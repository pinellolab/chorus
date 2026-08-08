"""Emit the reference REGION and SNP sets as a versioned, checksummed artefact.

Until now every build re-derived its positions from a seed at run time. That is
reproducible only as long as nothing upstream moves -- and the annotation GTF, the cCRE
BED, the DHS index and the FASTA are all inputs that can be updated without anyone
noticing that every background's reference class changed with them. Since the mixture
composition IS the reference-class definition (a percentile has no meaning apart from the
population it ranks against), a silent change there silently redefines every percentile
chorus reports.

So the positions become an artefact:

  * **SNP set** -- the reference population for the EFFECT null. "Is this variant's effect
    unusual among variants in comparable regulatory regions?"
  * **REGION set** -- the reference population for the ACTIVITY null (summary/perbin).
    "Is this locus active for this track, genome-wide?"

They are deliberately different populations; see docs/BACKGROUND_NULL_PROTOCOL.md §1.

What this buys, concretely:

  * a new oracle can be built against the IDENTICAL positions, so its percentiles are
    comparable with the existing eight rather than merely similar;
  * a rebuild is reproducible from the artefact even if an annotation source is updated,
    and the update becomes a deliberate, versioned act;
  * the content sha256 makes "which reference class is this background?" answerable from
    the file rather than from a build log.

It must REPRODUCE what the shipped builds used. ``--verify`` checks the realised strata
counts against a built oracle's provenance, and the generator replicates each builder's
exact call sequence (``random.seed(42)`` before iterating, ref base from the FASTA, alt
chosen from the three others in position order) rather than re-implementing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Which reference SNP set each oracle's EFFECT null is drawn from. Three families,
# because the right reference class depends on what the model predicts -- see
# docs/BACKGROUND_NULL_PROTOCOL.md section 3.
ORACLE_SNP_SET = {
    "enformer": "gene_anchored", "borzoi": "gene_anchored",
    "alphagenome": "gene_anchored", "sei": "gene_anchored",
    "epinformerseq": "gene_anchored",
    "legnet": "promoter",
    "chrombpnet": "accessibility", "cherimoya": "accessibility",
}
FAI = REPO / "genomes" / "hg38.fa.fai"
FASTA = REPO / "genomes" / "hg38.fa"


def _sha256_of(pairs) -> str:
    """Content hash over the sorted, canonical tuples. Order-independent by design."""
    h = hashlib.sha256()
    for row in sorted(pairs):
        h.update(("|".join(str(x) for x in row) + "\n").encode())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def _positions_to_snps(sampled, fasta, *, seed: int = 42):
    """Replicate the builders' position -> SNP step exactly.

    ``random.seed(seed)`` then iterate IN ORDER, reading the reference base and drawing
    the alt from the three others. The alt therefore depends on iteration order and on
    how many positions were skipped, so this cannot be reordered or vectorised without
    changing which SNPs are produced.
    """
    import pyfaidx

    fa = pyfaidx.Fasta(str(fasta), as_raw=True, sequence_always_upper=True)
    random.seed(seed)
    rows, skipped = [], 0
    for chrom, pos, stratum in sampled:
        ref_base = fa[chrom][pos - 1:pos]
        if ref_base not in "ACGT":
            skipped += 1
            continue
        alt = random.choice([b for b in "ACGT" if b != ref_base])
        rows.append((chrom, int(pos), ref_base, alt, stratum))
    return rows, skipped


def build(out_path: Path) -> dict:
    from chorus.utils.annotations import (
        DEFAULT_N_EFFECT_POSITIONS,
        DEFAULT_REGION_STRATA,
        PROMOTER_REGION_STRATA,
        load_chrom_sizes,
        sample_gene_anchored_positions,
        sample_promoter_anchored_positions,
    )

    sizes = load_chrom_sizes(str(FAI))
    n = DEFAULT_N_EFFECT_POSITIONS
    arrays: dict = {}
    prov: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/build_reference_position_sets.py",
        "generator_git_sha": _git_sha(),
        "genome": "hg38",
        "fai_sha256": _file_sha256(FAI),
        # The FASTA is 3.1 GB; hash the first 64 MB, which is enough to detect a
        # different assembly build without a five-minute read.
        "fasta_sha256_prefix64mb": _file_sha256(FASTA, limit=64 << 20),
        "n_effect_positions": n,
        "seeds": {"regions": 42, "dhs_pool": 43, "snp_alt": 42},
        "sets": {},
    }

    # ---- SNP sets (the EFFECT null's reference population) -------------------
    for name, sampler, strata in (
        ("gene_anchored", sample_gene_anchored_positions, DEFAULT_REGION_STRATA),
        ("promoter", sample_promoter_anchored_positions, PROMOTER_REGION_STRATA),
    ):
        logger.info("sampling SNP set %r (n=%d)", name, n)
        sampled = sampler(n, chrom_sizes=sizes, seed=42)
        rows, skipped = _positions_to_snps(sampled, FASTA)
        arrays[f"snps_{name}"] = np.array(
            [(c, p, r, a, s) for c, p, r, a, s in rows],
            dtype=[("chrom", "U32"), ("pos", "i8"), ("ref", "U1"),
                   ("alt", "U1"), ("stratum", "U16")])
        realised: dict = {}
        for _c, _p, _r, _a, s in rows:
            realised[s] = realised.get(s, 0) + 1
        prov["sets"][f"snps_{name}"] = {
            "kind": "snp",
            "purpose": "reference population for the EFFECT null",
            "n_positions_requested": n,
            "n_snps": len(rows),
            "n_skipped_non_acgt": skipped,
            "strata_requested": {k: round(v, 6) for k, v in strata.items()},
            "strata_realised": realised,
            "sha256": _sha256_of((c, p, r, a, s) for c, p, r, a, s in rows),
        }
        logger.info("  %s: %d SNPs, %d skipped, strata %s",
                    name, len(rows), skipped, realised)

    # ---- accessibility SNP set (ChromBPNet / Cherimoya) ---------------------
    # Replicates their inline construction exactly: 10,000 uniform SNPs (seed 42,
    # per-chromosome, 5 Mb margin, capped at 200 Mb) UNION 10,000 DHS-summit SNPs
    # (summits seed 43, +-150 bp jitter seed 44). Order and seeding matter: the alt
    # allele is drawn from the same global RNG stream as the positions.
    import pyfaidx
    from chorus.utils.annotations import sample_dhs_positions

    fa = pyfaidx.Fasta(str(FASTA), as_raw=True, sequence_always_upper=True)
    n_acc = 10_000
    logger.info("sampling SNP set 'accessibility' (%d random + %d DHS)", n_acc, n_acc)

    rows: list = []
    random.seed(42)
    chroms = [f"chr{i}" for i in range(1, 23)]
    per_chrom = n_acc // len(chroms) + 1
    for chrom in chroms:
        clen = len(fa[chrom])
        max_pos = min(clen - 5_000_000, 200_000_000)
        for _ in range(per_chrom):
            if len(rows) >= n_acc:
                break
            pos = random.randint(5_000_000, max_pos)
            rb = fa[chrom][pos - 1:pos]
            if rb not in "ACGT":
                continue
            rows.append((chrom, int(pos), rb,
                         random.choice([b for b in "ACGT" if b != rb]), "random"))
    random.shuffle(rows)
    rows = rows[:n_acc]
    n_random = len(rows)

    dhs = sample_dhs_positions(n_acc, seed=43)
    random.seed(44)
    n_dhs = 0
    for chrom, summit in dhs:
        pos = summit + random.randint(-150, 150)
        clen = len(fa[chrom])
        if pos < 5_000_000 or pos > clen - 5_000_000:
            continue
        rb = fa[chrom][pos - 1:pos]
        if rb not in "ACGT":
            continue
        rows.append((chrom, int(pos), rb,
                     random.choice([b for b in "ACGT" if b != rb]), "dhs"))
        n_dhs += 1

    arrays["snps_accessibility"] = np.array(
        rows, dtype=[("chrom", "U32"), ("pos", "i8"), ("ref", "U1"),
                     ("alt", "U1"), ("stratum", "U16")])
    prov["sets"]["snps_accessibility"] = {
        "kind": "snp",
        "purpose": "reference population for the EFFECT null (accessibility oracles)",
        "n_snps": len(rows),
        "strata_realised": {"random": n_random, "dhs": n_dhs},
        "note": "random 10,000 union DHS-summit 10,000, jittered +-150 bp; the "
                "composition ChromBPNet and Cherimoya have always used and which the "
                "2026-08 rebuild deliberately did NOT change",
        "sha256": _sha256_of(rows),
    }
    logger.info("  accessibility: %d SNPs (%d random + %d DHS)",
                len(rows), n_random, n_dhs)

    # ---- REGION set (the ACTIVITY null's reference population) ---------------
    # Genome-DOMINATED, unlike the SNP sets, because the question is "is this locus
    # active for this track, genome-wide?" -- most of the genome is silent for most
    # tracks, and that has to dominate the CDF so real peaks land as high percentiles.
    # Replicates the enformer/borzoi/alphagenome baseline construction exactly, including
    # its four separate RNG streams (789 random, 456 cCRE, 111 TSS, 222 gene body) and its
    # 10 Mb random-position margin, which is WIDER than the SNP sets' 5 Mb.
    from chorus.utils.annotations import get_annotation_manager, sample_ccre_positions

    logger.info("sampling REGION set 'genome_dominated'")
    region_rows: list = []

    n_random = 15_000
    random.seed(789)
    for chrom in [f"chr{i}" for i in range(1, 23)]:
        clen = len(fa[chrom])
        max_pos = min(clen - 10_000_000, 200_000_000)
        if max_pos <= 10_000_000:
            max_pos = clen - 1_000_000
        for _ in range(n_random // 22 + 1):
            if sum(1 for r in region_rows if r[2] == "random") >= n_random:
                break
            region_rows.append((chrom, random.randint(10_000_000, max_pos), "random"))

    for c, pos in sample_ccre_positions(
            n_per_category={"PLS": 3000, "dELS": 2500, "pELS": 1500, "CA-CTCF": 1500,
                            "CA-TF": 1000, "TF": 500, "CA-H3K4me3": 1000, "CA": 500},
            seed=456):
        region_rows.append((c, int(pos), "ccre"))

    mgr = get_annotation_manager()
    genes = mgr._get_genes_df(mgr.get_annotation_path("gencode_v48_basic"))
    pc = genes[genes["gene_type"] == "protein_coding"].copy()
    pc["tss"] = pc.apply(lambda r: r["start"] if r["strand"] == "+" else r["end"], axis=1)
    pc = pc[pc["chrom"].isin({f"chr{i}" for i in range(1, 23)})]
    tss = list(zip(pc.groupby("gene_name").first().reset_index()["chrom"],
                   pc.groupby("gene_name").first().reset_index()["tss"]))
    if len(tss) > 3000:
        tss = random.Random(111).sample(tss, 3000)
    for c, pos in tss:
        region_rows.append((str(c), int(pos), "tss"))

    long_g = pc[(pc["end"] - pc["start"]) > 10_000].copy()
    long_g["midpoint"] = (long_g["start"] + long_g["end"]) // 2
    gb = list(zip(long_g.groupby("gene_name").first().reset_index()["chrom"],
                  long_g.groupby("gene_name").first().reset_index()["midpoint"]))
    if len(gb) > 2000:
        gb = random.Random(222).sample(gb, 2000)
    for c, pos in gb:
        region_rows.append((str(c), int(pos), "gene_body"))

    arrays["regions_genome_dominated"] = np.array(
        region_rows, dtype=[("chrom", "U32"), ("pos", "i8"), ("stratum", "U16")])
    realised_r: dict = {}
    for _c, _p, s in region_rows:
        realised_r[s] = realised_r.get(s, 0) + 1
    prov["sets"]["regions_genome_dominated"] = {
        "kind": "region",
        "purpose": "reference population for the ACTIVITY null (summary + perbin)",
        "n_positions": len(region_rows),
        "strata_realised": realised_r,
        "note": "genome-dominated on purpose and NOT interchangeable with the SNP sets: "
                "if the two populations were unified, the acceptance criterion 'median "
                "activity percentile of the effect null's REF windows' would be 0.5 "
                "identically, for any track",
        "seeds": {"random": 789, "ccre": 456, "tss": 111, "gene_body": 222},
        "sha256": _sha256_of(region_rows),
    }
    logger.info("  regions: %d positions %s", len(region_rows), realised_r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov_json = json.dumps(prov, indent=1, sort_keys=True)
    np.savez_compressed(out_path, provenance=np.array([prov_json]), **arrays)
    logger.info("wrote %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)
    return prov


def verify(out_path: Path, oracle: str, bg_dir: "Path | None" = None) -> int:
    """Check the artefact reproduces what a BUILT oracle actually sampled."""
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    # Default to the LIVE dir, but a staged rebuild has to be checkable before it is
    # swapped -- that is the whole point of staging. Pointing this at the live files
    # while a rebuild is staged reports a 33% shortfall, which is correct (the shipped
    # backgrounds use the older 12,000-position class) and not what you meant to ask.
    bg_root = Path(bg_dir) if bg_dir else CHORUS_BACKGROUNDS_DIR
    logger.info("comparing against %s", bg_root)

    with np.load(out_path, allow_pickle=False) as d:
        prov = json.loads(str(d["provenance"][0]))
        family = ORACLE_SNP_SET.get(oracle)
        if family is None:
            logger.error("%s is not in ORACLE_SNP_SET; add it", oracle)
            return 1
        key = f"snps_{family}"
        if key not in d.files:
            logger.error("reference set has no %r (families: %s)", key,
                         [k for k in d.files if k.startswith("snps_")])
            return 1
        snps = d[key]
        logger.info("%s draws from the %r family", oracle, family)

    bg = bg_root / f"{oracle}_pertrack.npz"
    if not bg.exists():
        alt = bg_root / f"{oracle}_effect_cdfs_interim.npz"
        if alt.exists():
            bg = alt
            logger.info("  no pertrack yet; using the effect interim")
    if not bg.exists():
        logger.error("no built background for %s", oracle)
        return 1
    with np.load(bg, allow_pickle=True) as d:
        if "build_config" not in d.files:
            logger.warning("%s has no build_config; comparing counts only", oracle)
            cfg = {}
        else:
            cfg = json.loads(str(d["build_config"][0]))
        counts = d["effect_counts"]

    problems = []
    n_ref = len(snps)
    n_min = int(counts.min())
    logger.info("reference SNP set: %d SNPs", n_ref)
    logger.info("%s effect_counts: min=%d max=%d (%d distinct)",
                oracle, n_min, int(counts.max()), len(set(counts.tolist())))

    # The SNP SET is shared; the retained SUBSET is oracle-specific, because a window
    # whose N content exceeds max_n_fraction is rejected and window sizes differ by an
    # order of magnitude (Sei 4 kb, Enformer 393 kb, Borzoi 524 kb, AlphaGenome 1 Mb).
    # Measured: Sei kept all 17,909; Enformer 17,907; Borzoi 17,908. So percentiles across
    # oracles are computed against NEARLY the same population, not identically the same,
    # and the tolerance below is what makes that checkable rather than assumed.
    shortfall = n_ref - n_min
    if shortfall < 0:
        problems.append(
            f"{oracle} offered {n_min} > the reference set's {n_ref} SNPs, so it did not "
            f"use this set (or a layer fans out; check per-layer counts)")
    elif shortfall > max(10, int(0.001 * n_ref)):
        problems.append(
            f"{oracle} offered {n_min} against {n_ref} reference SNPs -- a shortfall of "
            f"{shortfall} ({shortfall / n_ref:.2%}), too large for window-level N "
            f"rejection. The built background probably used a different position set.")
    else:
        logger.info("  shortfall %d (%.3f%%) -- consistent with window-level N rejection",
                    shortfall, 100 * shortfall / n_ref)

    # Uniform counts are the healthy shape. A tight run of CONSECUTIVE integers is the
    # #123 partial-credit fingerprint: a per-variant exception mid-loop, so later tracks
    # saw fewer samples than earlier ones.
    distinct = sorted(set(counts.tolist()))
    if 1 < len(distinct) <= 8 and all(b - a == 1 for a, b in zip(distinct, distinct[1:])):
        problems.append(
            f"{oracle} effect_counts form a consecutive run {distinct} -- the #123 "
            f"partial-credit fingerprint, not a fan-out")
    logged = (cfg.get("effect_region_set_as_logged") or {}).get("strata_counts")
    if logged:
        ref_strata = prov["sets"][key]["strata_realised"]
        if {k: int(v) for k, v in logged.items()} != {k: int(v) for k, v in ref_strata.items()}:
            problems.append(
                f"{oracle} logged strata {logged} != reference {ref_strata}")
        else:
            logger.info("strata match the build log exactly: %s", ref_strata)
    for p in problems:
        logger.error(p)
    if problems:
        return 1
    logger.info("VERIFIED: the reference set reproduces %s's sampled population", oracle)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "reference_sets"
                                        / "chorus_reference_positions_v1.npz"))
    ap.add_argument("--verify-against", default=None,
                    help="an oracle whose built background must match this set")
    ap.add_argument("--backgrounds-dir", default=None,
                    help="where to look for that oracle's background; defaults to the "
                         "live CHORUS_BACKGROUNDS_DIR. Point it at a staging directory "
                         "to check a rebuild BEFORE it is swapped.")
    args = ap.parse_args()
    out = Path(args.out)
    if args.verify_against:
        return verify(out, args.verify_against, args.backgrounds_dir)
    build(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
