"""Chunked, resumable, single-flight HTTP downloader.

Chorus fetches several large artifacts from the public internet (hg38
reference genome ~938 MB, SEI Zenodo tarball ~3.2 GB, 24 ChromBPNet ENCODE
tarballs ~270-700 MB each). The stdlib convenience wrapper
``urllib.request.urlretrieve`` observed throughput as low as ~80 KB/s on
macOS (see audits/2026-04-14_macos_arm64.md) and has no resume path — an
interrupted download starts from zero, which for the SEI tar alone is
~11 hours at that rate.

``download_with_resume`` replaces it with a plain stdlib ``urlopen`` chunked
read loop that:

* Resumes a previous partial via the ``Range: bytes=<offset>-`` request
  header when ``dest.partial`` already exists.
* Holds an exclusive ``fcntl.flock`` on ``dest.lock`` so two concurrent
  callers (e.g. ``chorus health`` racing a manual ``create_oracle(...)``)
  cannot overwrite each other's partial file.
* Logs progress every ~100 MB so long downloads are visibly alive.

No new runtime deps — stdlib only.
"""
from __future__ import annotations

import fcntl
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_with_resume(
    url: str,
    dest: str | Path,
    chunk_bytes: int = 4 * 1024 * 1024,
    label: Optional[str] = None,
    log_every_bytes: int = 100 * 1024 * 1024,
) -> None:
    """Streamed HTTP GET with ``Range`` resume + single-flight lock.

    Args:
        url: Source URL.
        dest: Final destination path. A sibling ``<dest>.partial`` is used
            during download and ``<dest>.lock`` for the single-flight lock.
        chunk_bytes: Read buffer size. 4 MiB is a reasonable default.
        label: Human-readable name used in progress logs. Defaults to the
            destination basename.
        log_every_bytes: Emit a progress log line at most every N bytes.
            Default 100 MiB so per-download log spam stays bounded.

    Notes:
        * If ``dest`` already exists the function returns immediately.
        * If the server replies 416 (Range Not Satisfiable) on a resume
          request the partial is promoted to the final destination — means
          we already had everything.
        * If the server ignores ``Range`` and returns 200 the partial is
          truncated and the download restarts from byte 0.
    """
    dest_p = Path(dest)
    partial_p = dest_p.with_suffix(dest_p.suffix + ".partial")
    lock_p = dest_p.with_suffix(dest_p.suffix + ".lock")
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tag = label or dest_p.name

    with open(lock_p, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            # Another process may have finished while we waited for the lock.
            if dest_p.exists():
                return

            already = partial_p.stat().st_size if partial_p.exists() else 0
            req = urllib.request.Request(url)
            if already > 0:
                req.add_header("Range", f"bytes={already}-")
                logger.info("Resuming %s at byte %s", tag, f"{already:,}")

            try:
                resp = urllib.request.urlopen(req, timeout=60)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and already > 0:
                    partial_p.rename(dest_p)
                    return
                raise

            status = getattr(resp, "status", None) or resp.getcode()
            if already > 0 and status != 206:
                logger.warning("Server ignored Range header for %s; restarting", tag)
                already = 0
                partial_p.unlink(missing_ok=True)

            total = None
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                total = int(cl) + already

            mode = "ab" if already > 0 else "wb"
            downloaded = already
            last_log = downloaded
            pbar = tqdm(
                total=total,
                initial=already,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=tag,
                disable=None,  # auto-disable when stderr isn't a TTY
            )
            try:
                with open(partial_p, mode) as out:
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        # When tqdm is disabled (non-TTY, e.g. CI logs),
                        # emit a periodic log line so the download is
                        # still visibly alive.
                        if (
                            pbar.disable
                            and total
                            and downloaded - last_log >= log_every_bytes
                        ):
                            pct = 100.0 * downloaded / total
                            logger.info(
                                "  %s: %.2f/%.2f GB (%.1f%%)",
                                tag, downloaded / 1e9, total / 1e9, pct,
                            )
                            last_log = downloaded
            finally:
                pbar.close()

            partial_p.rename(dest_p)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            lock_p.unlink(missing_ok=True)
