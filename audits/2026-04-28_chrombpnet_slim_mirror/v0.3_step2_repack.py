"""Step 2 — fetch + repack remaining ChromBPNet cell-types into the slim layout.

For every (assay, ENCFF) in chorus's ChromBPNet registry that doesn't yet
have a slim entry under /tmp/hf-staging/, parallel-download the ENCODE
tarball (resumable), extract just `models/fold_0/model.chrombpnet_nobias.
fold_0.*.h5`, copy it into the slim layout, and write/update
manifest.json.

After this step, /tmp/hf-staging/{ATAC,DNASE}/<canonical_cell_type>/
fold_0/model.chrombpnet_nobias.fold_0.<ENCSR>.h5 should exist for all
42 unique ChromBPNet annotations.

Layout (per the plan):
    /tmp/hf-staging/
        DNASE/HepG2/fold_0/model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5
        DNASE/K562/fold_0/model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5
        ATAC/K562/fold_0/model.chrombpnet_nobias.fold_0.ENCSR868FGK.h5
        ...
        manifest.json

manifest.json keys: ENCFF accession.
manifest.json value per entry: sha256, size_bytes, slim_path, assay,
canonical_cell_type, source_encsr (parsed from extracted h5 filename).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/Users/lp698/chorus_test/chorus")
sys.path.insert(0, str(REPO))

from chorus.oracles.chrombpnet_source.chrombpnet_globals import (  # noqa: E402
    iter_unique_models,
)
from chorus.utils.http import download_with_resume  # noqa: E402

STAGING = Path("/tmp/hf-staging")
TARBALL_CACHE = REPO / "downloads/chrombpnet"
MANIFEST = STAGING / "manifest.json"

# ENCFF for retina DNase-seq has no model file yet (commented out in
# chrombpnet_globals.py); skip it gracefully if encountered.
SKIP_ENCFF: set[str] = set()

NOBIAS_PATTERN = re.compile(r"(?:^|/)fold_0/model\.chrombpnet_nobias\.fold_0\.(ENCSR[A-Z0-9]+)\.h5$")


def encode_url(encff: str) -> str:
    return f"https://www.encodeproject.org/files/{encff}/@@download/{encff}.tar.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim_path(assay: str, cell_type: str, encsr: str) -> Path:
    return STAGING / assay / cell_type / "fold_0" / f"model.chrombpnet_nobias.fold_0.{encsr}.h5"


def already_slim(assay: str, cell_type: str) -> Path | None:
    """Return the slim h5 if already extracted, else None."""
    d = STAGING / assay / cell_type / "fold_0"
    if not d.exists():
        return None
    matches = list(d.glob("model.chrombpnet_nobias.fold_0.*.h5"))
    return matches[0] if matches else None


def extract_nobias(tarball: Path, assay: str, cell_type: str) -> Path:
    """Extract only the fold_0 nobias h5 from the tarball into the slim
    layout. Returns the slim path. ENCODE tarballs have ./fold_<N>/
    paths at the root (no leading models/ prefix); we search the TOC
    for the nobias fold_0 h5 directly."""
    with tarfile.open(tarball, "r:gz") as tf:
        target_member = None
        target_encsr = None
        for member in tf:
            m = NOBIAS_PATTERN.search(member.name)
            if m:
                target_member = member
                target_encsr = m.group(1)
                break
        if target_member is None:
            raise RuntimeError(
                f"No fold_0/model.chrombpnet_nobias.fold_0.*.h5 in {tarball}"
            )

        out = slim_path(assay, cell_type, target_encsr)
        out.parent.mkdir(parents=True, exist_ok=True)
        ef = tf.extractfile(target_member)
        if ef is None:
            raise RuntimeError(f"Could not extract {target_member.name} from {tarball}")
        with open(out, "wb") as fout:
            shutil.copyfileobj(ef, fout, length=1 << 20)
    return out


def fetch_and_repack(assay: str, cell_type: str, encff: str) -> dict:
    """Download ENCODE tarball (or use cached) and repack to slim layout."""
    if encff in SKIP_ENCFF:
        return {"encff": encff, "status": "SKIP", "reason": "no model file"}

    # Already in slim layout?
    existing = already_slim(assay, cell_type)
    if existing is not None:
        return {"encff": encff, "status": "ALREADY_SLIM", "slim_path": str(existing)}

    # Cached tarball available? Check both standard chorus layout and
    # the scratch dir from a prior run of this script.
    cached_tarball = None
    for candidate_dir in (
        TARBALL_CACHE / f"{assay}_{cell_type}",
        TARBALL_CACHE / f"_slim_repack_{assay}_{cell_type}",
    ):
        candidate = candidate_dir / f"{encff}.tar.gz"
        if candidate.exists() and candidate.stat().st_size > 600_000_000:
            cached_tarball = candidate
            break

    # Otherwise fetch into a scratch dir
    scratch = TARBALL_CACHE / f"_slim_repack_{assay}_{cell_type}"
    scratch.mkdir(parents=True, exist_ok=True)
    tarball = cached_tarball or (scratch / f"{encff}.tar.gz")
    if cached_tarball is None:
        download_with_resume(encode_url(encff), tarball, label=f"{assay}:{cell_type}")

    # Extract nobias h5
    slim = extract_nobias(tarball, assay, cell_type)

    # Clean up scratch download (keep cached tarballs in their original location)
    if cached_tarball is None and tarball.exists():
        tarball.unlink()
        try:
            scratch.rmdir()
        except OSError:
            pass

    return {
        "encff": encff,
        "assay": assay,
        "cell_type": cell_type,
        "slim_path": str(slim),
        "status": "OK",
    }


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)

    # Build deduped work list (ENCFF as key)
    work: dict[str, tuple[str, str, str]] = {}
    for assay, cell_type, encff in iter_unique_models():
        if encff in SKIP_ENCFF:
            continue
        # Prefer canonical (bare-biosample) cell_type names; iter_unique_models()
        # already returns the canonical name per ENCFF.
        if encff not in work:
            work[encff] = (assay, cell_type, encff)
    print(f"Work list: {len(work)} unique ENCFFs to ensure slim coverage for")

    # Run extractions in parallel — modest concurrency to be kind to ENCODE.
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_and_repack, *w): w for w in work.values()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as exc:
                a, c, e = futs[fut]
                res = {"encff": e, "assay": a, "cell_type": c, "status": "FAIL", "error": str(exc)}
            print(f"  [{i}/{len(work)}] {res.get('assay','?')}:{res.get('cell_type','?')} ({res['encff']}) → {res['status']}")
            results.append(res)

    # Compute sha256 + size + write manifest for everything in slim
    print("\nComputing sha256 + writing manifest ...")
    manifest: dict[str, dict] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    for r in results:
        if r["status"] not in ("OK", "ALREADY_SLIM"):
            continue
        slim = Path(r["slim_path"])
        encsr = slim.stem.split(".")[-1]  # last component
        manifest[r["encff"]] = {
            "sha256": sha256_file(slim),
            "size_bytes": slim.stat().st_size,
            "slim_path": str(slim.relative_to(STAGING)),
            "assay": r["assay"],
            "canonical_cell_type": r["cell_type"],
            "source_encsr": encsr,
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {len(manifest)} entries → {MANIFEST}")
    print(f"slim total size: {sum(m['size_bytes'] for m in manifest.values()) / 1e6:.1f} MB")

    fails = [r for r in results if r["status"] == "FAIL"]
    if fails:
        print(f"\nFAIL: {len(fails)} entries failed")
        for r in fails:
            print(f"  {r['encff']}: {r.get('error','?')}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
