"""Vendor the BATv1 metadata and fold-0 validation metrics into ``cherimoya_source/``.

CATv1's two TSVs were copied verbatim from its Hugging Face repository. BATv1's
metadata table is six times larger (1.3 MB, 31 columns), and twenty of its columns
record the control-file md5s and ENCODE portal statuses that the release needs but the
oracle never reads. This script fetches both release files at the revision pinned in
``batv1_globals.py`` and writes the columns chorus uses:

* ``BATv1-metadata.tsv`` -- CATv1's ten columns (with ``experiment_perturbed`` spelled
  correctly, as the release spells it), plus the ChIP-seq target, ENCODE's classes for it,
  the experiment's portal status, the peak count, and the release's ``quality_flags``.
* ``BATv1-performance-fold0.tsv`` -- the fold-0 rows of ``performance.validation.tsv``,
  in the same six columns as ``CATv1-performance-fold0.tsv``.

Run from the repo root, then regenerate the defaults table:

    python scripts/vendor_batv1_metadata.py
    python scripts/generate_batv1_defaults.py
"""

import io
import sys
from pathlib import Path

import pandas
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chorus.oracles.cherimoya_source.batv1_globals import (  # noqa: E402
    BATV1_HF_REPO,
    BATV1_HF_REVISION,
)

SOURCE_DIR = REPO_ROOT / "chorus" / "oracles" / "cherimoya_source"

METADATA_COLUMNS = [
    "experiment_accession",
    "annotation_accession",
    "assay_term_name",
    "experiment_perturbed",
    "experiment_biosample_term_name",
    "assembly",
    "organism",
    "biosample_simple_summary",
    "biosample_classification",
    "biosample_term_id",
    "target_label",
    "target_investigated_as",
    "experiment_status",
    "n_peaks",
    "quality_flags",
]

PERFORMANCE_COLUMNS = [
    "experiment_accession",
    "fold",
    "profile_pearson",
    "profile_spearman",
    "count_pearson",
    "count_spearman",
]


def _fetch(filename: str) -> str:
    url = f"https://huggingface.co/{BATV1_HF_REPO}/resolve/{BATV1_HF_REVISION}/{filename}"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.text


def main() -> int:
    meta = pandas.read_csv(io.StringIO(_fetch("BATv1-metadata.tsv")), sep="\t", dtype=str)
    perf = pandas.read_csv(io.StringIO(_fetch("performance.validation.tsv")), sep="\t")

    missing = [c for c in METADATA_COLUMNS if c not in meta.columns]
    if missing:
        raise SystemExit(f"BATv1-metadata.tsv lacks columns {missing}")
    fold0 = perf[perf["fold"] == 0]
    if len(fold0) != len(meta):
        raise SystemExit(
            f"{len(fold0)} fold-0 performance rows for {len(meta)} experiments"
        )

    meta[METADATA_COLUMNS].to_csv(SOURCE_DIR / "BATv1-metadata.tsv", sep="\t", index=False)
    # Five decimals, as CATv1-performance-fold0.tsv carries; the release file's
    # float64 repr would double the vendored size for no information.
    fold0[PERFORMANCE_COLUMNS].to_csv(
        SOURCE_DIR / "BATv1-performance-fold0.tsv", sep="\t", index=False, float_format="%.5f"
    )
    print(f"wrote {len(meta)} experiments at revision {BATV1_HF_REVISION[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
