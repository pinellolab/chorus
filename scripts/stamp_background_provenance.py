"""Stamp as-built provenance onto the three rebuilt backgrounds (#124).

``append_tracks`` and ``build_and_save`` learned to carry a file-level
``build_config`` in the provenance PR, but none of the three rebuild-set builders
passes one — so alphagenome, borzoi and enformer shipped with only the canonical 8
keys. Cherimoya's builder writes its own and is the shape to match.

Re-running ~11 h of forward passes to add metadata would be absurd, and it is not
necessary: ``build_config`` is file-level, so it can be appended in place. What
matters is that **only establishable facts go in**.

Specifically NOT stamped:

* ``xla_flags`` — reading ``os.environ`` here would record THIS process's flags,
  not the build's, which is fabricated provenance. ``pin_deterministic_xla_flags``
  logs nothing on its success path, so the build logs carry no positive record
  either. Determinism is instead asserted empirically by
  ``scripts/gate_end_to_end_determinism.py``.
* a bare ``builder_git_sha`` set to today's ``HEAD`` — for the same reason. Two
  commits landed after these builds finished, so HEAD is *not* what ran, and
  recording it would read as authoritative while being wrong. A commit sha also
  says nothing about whether the working tree matched it, and the AlphaGenome build
  in fact started at 16:26 with its builder changes still UNCOMMITTED (they became
  c4ded13 at 19:28, three hours into an eleven-hour run).
* wall-clock or throughput figures, for the same reason.

Instead the stamp is content-addressed: a ``git hash-object`` sha1 per source file
plus that file's mtime and a boolean for whether the mtime predates the build
start. Python reads source at process start, so ``unchanged_since_build_start:
true`` means the recorded hash IS what ran — a checkable claim rather than an
implied one. Where it is false, the stamp says so.

Everything else is read from the builder source, from the shipped array shapes, or
hashed from the FASTA on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Resolved through the data-dir mechanism, not hardcoded to $HOME. Every
# background-handling script had this literal; CHORUS_BACKGROUNDS_DIR applies
# the legacy ~/.chorus compatibility itself, per kind.
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
BG = CHORUS_BACKGROUNDS_DIR
FASTA = REPO / "genomes" / "hg38.fa"

# Build windows, read from the actual run logs. These let the stamp state a
# VERIFIABLE fact — whether each source file's mtime predates the build start, and
# so whether its current content is what ran — instead of recording today's HEAD
# and implying the build came from it. Two commits landed after these finished.
BUILD_WINDOWS = {
    "alphagenome": ("2026-08-03T16:26:14", "2026-08-04T03:48:43",
                    "b07ab9c", "logs/bg_alphagenome_variants.log"),
    "borzoi": ("2026-08-03T03:08:07", "2026-08-03T07:24:46",
               "b07ab9c", "/data/chorus_data/rebuild_borzoi_variants.log"),
    "enformer": ("2026-08-03T03:08:12", "2026-08-03T07:17:41",
                 "b07ab9c", "/data/chorus_data/rebuild_enformer_variants.log"),
}

# The files whose content determines a sampled value. background_sampling.py is
# included even though its guard changed afterwards, precisely so the stamp says
# so rather than hiding it.
SOURCE_FILES = [
    "chorus/utils/annotations.py",
    "chorus/analysis/background_sampling.py",
    "chorus/analysis/normalization.py",
    "chorus/analysis/scorers.py",
]

# Read from each builder's source, not guessed. INPUT_LENGTH is the grep above;
# resolution and window are the values the shared helpers are called with.
BUILDERS = {
    "alphagenome": dict(
        input_length=1_048_576, resolution=1,
        rna_aggregation="mean over the gene's merged exon mask in bins, ln, "
                        "pseudocount 1e-3 (AlphaGenome GeneMaskLFCScorer)",
        rna_gene_selection="TSS-in-window, semi-open, per-transcript, strand-aware "
                           "(AlphaGenome gene_mask_extractor); protein_coding only",
        cage_aggregation="sum over a 501 bp window centred on the variant, log2, "
                         "pseudocount 1.0 (AlphaGenome CenterMaskScorer "
                         "DIFF_LOG2_SUM)",
        histone_window_bp=2001, other_window_bp=501,
    ),
    "borzoi": dict(
        input_length=524_288, resolution=32,
        rna_aggregation="mean over the gene's merged exon bins, log2, pseudocount 1.0",
        rna_gene_selection="overlap rule (gene span intersects the window)",
        cage_aggregation="sum over a 501 bp window centred on the variant, log2, "
                         "pseudocount 1.0",
        histone_window_bp=2001, other_window_bp=501,
    ),
    "enformer": dict(
        input_length=393_216, resolution=128,
        rna_aggregation=None,          # enformer has no RNA layer
        rna_gene_selection=None,
        cage_aggregation="sum over a 501 bp window centred on the variant, log2, "
                         "pseudocount 1.0",
        histone_window_bp=2001, other_window_bp=501,
    ),
}


def _git_blob(path: Path) -> str:
    """Content hash of the file AS IT IS NOW — exact and independently checkable
    with ``git hash-object``, unlike a commit sha which says nothing about whether
    the working tree matched it."""
    return subprocess.run(["git", "hash-object", str(path)], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def _source_state(rel: str, build_start: str) -> dict:
    path = REPO / rel
    if not path.exists():
        return {"present": False}
    from datetime import datetime
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return {
        "sha1": _git_blob(path),
        "mtime": mtime,
        # The load-bearing claim. Python reads source at process start, so if the
        # file was last modified before the build began, the hash above IS what ran.
        "unchanged_since_build_start": mtime < build_start,
    }


def _logged_region_set(log_rel: str) -> dict:
    """The strata the build ITSELF logged — the strongest provenance available.

    Better than any mtime or commit sha, because it is the running process
    reporting what it sampled. It settles a case the mtimes could not: borzoi and
    enformer started at 03:08:07 while the commit that introduced gene-anchored
    sampling (c10995a) landed at 03:16:51, nine minutes later, and their builder
    mtimes were then bumped to 03:30:14 by a git operation. The mtime comparison
    therefore reads "changed after build start" for both. The log line proves the
    gene-anchored code was in fact running from the first minute — as uncommitted
    working-tree edits, the same way AlphaGenome's were.
    """
    path = Path(log_rel) if Path(log_rel).is_absolute() else REPO / log_rel
    if not path.exists():
        return {"available": False, "reason": f"{log_rel} absent"}
    import ast
    import re
    pattern = re.compile(
        r"Generated (\d+) gene-anchored SNPs from (\d+) sampled positions: (\{.*\})")
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return {
                    "available": True,
                    "logged_at": line[:19],
                    "n_snps": int(m.group(1)),
                    "region_set": m.group(2),
                    "n_positions": int(m.group(3)),
                    "strata_counts": ast.literal_eval(m.group(4)),
                }
    return {"available": False, "reason": "no gene-anchored SNP line in the log"}


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracles", nargs="*", default=list(BUILDERS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing build_config")
    args = ap.parse_args()

    print(f"[prov] hashing {FASTA} ...", flush=True)
    fasta_sha = _sha256(FASTA)
    print(f"[prov] fasta sha256 {fasta_sha[:16]}...")

    # Measured, not assumed. This stamper is superseded by stamp_provenance_v4.py
    # (schema 2 is below MIN_ARTEFACT_SCHEMA, so nothing it writes would be
    # accepted today), but a stamper that can state an unchecked assembly is a
    # stamper that can lie, and it costs one call not to be.
    from chorus.utils.genome import detect_assembly
    genome = detect_assembly(FASTA)
    if genome is None:
        raise SystemExit(f"[prov] cannot identify the assembly of {FASTA}; "
                         f"refusing to stamp 'genome' as a guess")
    print(f"[prov] {FASTA} is {genome}")

    from chorus.utils.annotations import DEFAULT_REGION_STRATA

    for name in args.oracles:
        path = BG / f"{name}_pertrack.npz"
        if not path.exists():
            print(f"[prov] SKIP {name}: {path} absent")
            continue
        with np.load(path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}
        if "build_config" in payload and not args.force:
            print(f"[prov] SKIP {name}: already stamped (--force to replace)")
            continue

        spec = BUILDERS[name]
        start, finish, head, log = BUILD_WINDOWS[name]
        builder_rel = f"scripts/build_backgrounds_{name}.py"
        sources = {r: _source_state(r, start) for r in SOURCE_FILES}
        config = {
            "schema_version": 2,
            "oracle": name,
            "genome": genome,
            "fasta_sha256": fasta_sha,
            "build_started": start,
            "build_finished": finish,
            "build_log": log,
            "head_at_build_start": head,
            "builder_script": builder_rel,
            "builder_script_state": _source_state(builder_rel, start),
            "source_state": sources,
            "input_length": spec["input_length"],
            "resolution": spec["resolution"],
            "effect_region_strata": dict(DEFAULT_REGION_STRATA),
            "effect_region_rule": "gene-anchored, sampled per stratum from "
                                  "protein-coding annotation (GENCODE v48 basic)",
            # What the build actually logged, which outranks every mtime here.
            "effect_region_set_as_logged": _logged_region_set(log),
            "cage_aggregation": spec["cage_aggregation"],
            "rna_aggregation": spec["rna_aggregation"],
            "rna_gene_selection": spec["rna_gene_selection"],
            "histone_window_bp": spec["histone_window_bp"],
            "other_window_bp": spec["other_window_bp"],
            "cdf_points": int(payload["effect_cdfs"].shape[1]),
            "n_tracks": int(len(payload["track_ids"])),
            # Deliberately absent: xla_flags, wall_clock. See the module docstring —
            # this process's environment is not the build's, and inventing the
            # field would be worse than omitting it.
            "determinism": "asserted by scripts/gate_end_to_end_determinism.py, "
                           "not recorded here; recover flag behaviour from "
                           "chorus/core/determinism.py at builder_git_sha",
        }
        print(f"\n[prov] {name}: {json.dumps(config, indent=2)[:400]}...")
        if args.dry_run:
            continue
        payload["build_config"] = np.array([json.dumps(config)])
        # np.savez_compressed APPENDS ".npz" unless the name already ends in it,
        # so a ".stamping" suffix silently writes elsewhere and the verify below
        # reads a file that does not exist.
        tmp = path.with_name(path.name.replace(".npz", ".stamping.npz"))
        np.savez_compressed(tmp, **payload)
        # verify before replacing — a truncated write here loses a 10 h artefact
        with np.load(tmp, allow_pickle=True) as check:
            assert len(check["track_ids"]) == config["n_tracks"]
            assert json.loads(str(check["build_config"][0]))["oracle"] == name
            for key in ("effect_cdfs", "effect_counts"):
                assert np.array_equal(check[key], payload[key]), key
        tmp.replace(path)
        print(f"[prov] {name}: stamped, {path.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
