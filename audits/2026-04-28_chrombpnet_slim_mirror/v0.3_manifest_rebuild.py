"""Rebuild manifest.json from the on-disk slim layout.

Walks /tmp/hf-staging/ for every .h5, computes sha256 + size, and
writes manifest.json keyed by ENCFF (for ChromBPNet) or BP_BASE_ID
(for BPNet). For ChromBPNet entries we resolve back to (assay,
canonical_cell_type, ENCFF) via iter_unique_models() — every slim
h5 lives at {assay}/{canonical_cell_type}/fold_0/<filename>.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/lp698/chorus_test/chorus")
sys.path.insert(0, str(REPO))

from chorus.oracles.chrombpnet_source.chrombpnet_globals import (  # noqa: E402
    iter_unique_models, iter_unique_bpnet_models,
)

STAGING = Path("/tmp/hf-staging")
MANIFEST = STAGING / "manifest.json"

# canonical_cell_type → (assay, ENCFF) for ChromBPNet ATAC/DNase
CANON_TO_ENCFF: dict[tuple[str, str], str] = {}
for assay, cell_type, encff in iter_unique_models():
    CANON_TO_ENCFF[(assay, cell_type)] = encff

# BP_BASE_ID → (cell_type, tf, model_url)
BP_INFO: dict[str, dict] = {}
for cell_type, tf, url, identifier in iter_unique_bpnet_models():
    BP_INFO[identifier] = {"cell_line": cell_type, "tf": tf, "source_url": url}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest: dict[str, dict] = {}

    # ChromBPNet: {assay}/{cell_type}/fold_0/*.h5
    encsr_re = re.compile(r"\.fold_0\.(ENCSR[A-Z0-9]+)\.h5$")
    for assay in ("ATAC", "DNASE"):
        for h5 in (STAGING / assay).rglob("*.h5"):
            # Path: STAGING/<assay>/<cell_type>/fold_0/model.chrombpnet_nobias.fold_0.<ENCSR>.h5
            cell_type = h5.parent.parent.name
            encff = CANON_TO_ENCFF.get((assay, cell_type))
            if encff is None:
                print(f"  WARN: no ENCFF for ({assay}, {cell_type}); skip {h5}")
                continue
            m = encsr_re.search(h5.name)
            encsr = m.group(1) if m else None
            manifest[encff] = {
                "kind": "ChromBPNet",
                "assay": assay,
                "canonical_cell_type": cell_type,
                "source_encsr": encsr,
                "slim_path": str(h5.relative_to(STAGING)),
                "size_bytes": h5.stat().st_size,
                "sha256": sha256_file(h5),
            }

    # BPNet: BPNet/<BP_BASE_ID>.h5
    for h5 in (STAGING / "BPNet").glob("*.h5"):
        identifier = h5.stem  # e.g. BP000001.1
        info = BP_INFO.get(identifier, {})
        manifest[identifier] = {
            "kind": "BPNet",
            "cell_line": info.get("cell_line"),
            "tf": info.get("tf"),
            "source_url": info.get("source_url"),
            "slim_path": str(h5.relative_to(STAGING)),
            "size_bytes": h5.stat().st_size,
            "sha256": sha256_file(h5),
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    encff_count = sum(1 for k in manifest if k.startswith("ENCFF"))
    bp_count = sum(1 for k in manifest if k.startswith("BP"))
    encff_size = sum(m["size_bytes"] for k, m in manifest.items() if k.startswith("ENCFF"))
    bp_size = sum(m["size_bytes"] for k, m in manifest.items() if k.startswith("BP"))
    total = encff_size + bp_size
    print(f"manifest: {len(manifest)} entries → {MANIFEST}")
    print(f"  ChromBPNet ATAC/DNase ENCFF entries: {encff_count} ({encff_size/1e6:.1f} MB)")
    print(f"  BPNet/CHIP BP entries:               {bp_count} ({bp_size/1e6:.1f} MB)")
    print(f"  Total slim mirror size:              {total/1e6:.1f} MB ({total/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
