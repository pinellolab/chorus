"""Step 3 — bulk download 744 BPNet/CHIP h5 files into the slim layout.

Source: https://mencius.uio.no/JASPAR/JASPAR_DeepLearning/2026/models/BP*/
Each model.h5 is exactly 563,760 bytes per the plan's verified facts.

Slim layout: BPNet/{BASE_ID}.h5

After this step, /tmp/hf-staging/BPNet/ contains 744 h5 files
(BP000001.1.h5 ... BP001257.1.h5 minus a few unused IDs). The
manifest.json gets BP entries appended:
  {BASE_ID: {sha256, size_bytes, slim_path, source_url, tf_name, cell_line}}
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/Users/lp698/chorus_test/chorus")
sys.path.insert(0, str(REPO))

from chorus.oracles.chrombpnet_source.chrombpnet_globals import (  # noqa: E402
    iter_unique_bpnet_models,
)
from chorus.utils.http import download_with_resume  # noqa: E402

STAGING = Path("/tmp/hf-staging")
BPNET_DIR = STAGING / "BPNet"
MANIFEST = STAGING / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(cell_type: str, tf: str, model_url: str, identifier: str) -> dict:
    out = BPNET_DIR / f"{identifier}.h5"
    if out.exists() and out.stat().st_size > 500_000:
        return {
            "identifier": identifier,
            "tf": tf,
            "cell_type": cell_type,
            "model_url": model_url,
            "out": str(out),
            "status": "ALREADY_SLIM",
        }
    try:
        download_with_resume(model_url, out, label=f"BPNet:{cell_type}:{tf}")
        return {
            "identifier": identifier,
            "tf": tf,
            "cell_type": cell_type,
            "model_url": model_url,
            "out": str(out),
            "status": "OK",
        }
    except Exception as exc:
        return {
            "identifier": identifier,
            "tf": tf,
            "cell_type": cell_type,
            "model_url": model_url,
            "status": "FAIL",
            "error": str(exc),
        }


def main() -> int:
    BPNET_DIR.mkdir(parents=True, exist_ok=True)

    work = list(iter_unique_bpnet_models())  # 744 entries
    print(f"BPNet work list: {len(work)} models")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, *w): w for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["status"] == "FAIL" or i % 50 == 0 or i == len(work):
                print(f"  [{i}/{len(work)}] {res['cell_type']}:{res['tf']} ({res['identifier']}) → {res['status']}")
            results.append(res)

    # Update manifest with BP* entries
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for r in results:
        if r["status"] not in ("OK", "ALREADY_SLIM"):
            continue
        out = Path(r["out"])
        manifest[r["identifier"]] = {
            "sha256": sha256_file(out),
            "size_bytes": out.stat().st_size,
            "slim_path": str(out.relative_to(STAGING)),
            "source_url": r["model_url"],
            "tf": r["tf"],
            "cell_line": r["cell_type"],
            "kind": "BPNet",
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    bp_count = sum(1 for k in manifest if k.startswith("BP"))
    bp_size = sum(m["size_bytes"] for k, m in manifest.items() if k.startswith("BP"))
    print(f"\nBPNet entries in manifest: {bp_count}, total size {bp_size/1e6:.1f} MB")

    fails = [r for r in results if r["status"] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} BPNet downloads failed")
        for r in fails[:10]:
            print(f"  {r['identifier']}: {r.get('error','?')}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
