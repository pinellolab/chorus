"""Score rs12740374 with LegNet at the SNP site (NARROW window).

The wide-window canonical run averages LegNet across ~1 Mb and reports
~0 (the variant isn't at a promoter). But for a single-fragment MPRA
readout centered on the variant, we want the LOCAL prediction. This
matches asking 'if I cloned the 230 bp around rs12740374 into a
LentiMPRA construct, would alt drive more expression than ref?'.
"""
from __future__ import annotations
import json, logging, os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("legnet_narrow")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_verified_legnet_narrow.json")
GENOME = os.path.join(REPO_ROOT, "genomes/hg38.fa")
V = dict(chrom="chr1", position=109274968, ref="G", alt="T", id="rs12740374", gene="SORT1")

from chorus.oracles.legnet import LegNetOracle
from chorus.analysis.variant_report import build_variant_report
from chorus.analysis.analysis_request import AnalysisRequest

o = LegNetOracle(cell_type="HepG2", assay="LentiMPRA", use_environment=False)
o.load_pretrained_model()
log.info("LegNet sequence_length=%s", o.sequence_length)

# Narrow region: 2 bp around the variant. predict_variant_effect will
# pull the LegNet-window-sized context (~230 bp) centered on the SNP.
region = f"{V['chrom']}:{V['position']}-{V['position']+1}"
pos    = f"{V['chrom']}:{V['position']}"

r = o.predict_variant_effect(
    genomic_region=region, variant_position=pos,
    alleles=[V["ref"], V["alt"]],
    assay_ids=["LentiMPRA:HepG2"], genome=GENOME,
)
rep = build_variant_report(r, oracle_name="legnet",
                           gene_name=V["gene"], normalizer=None,
                           analysis_request=AnalysisRequest(
                               user_prompt="narrow LegNet at SNP site",
                               tool_name="x", oracle_name="legnet",
                               normalizer_name="(none)",
                               tracks_requested="LentiMPRA:HepG2"))
d = rep.to_dict()
with open(OUT, "w") as f:
    json.dump(d, f, indent=2, default=str)

for a, b in d.get("alleles", {}).items():
    for s in b.get("all_scores", []):
        log.info("  %s : raw=%s ref=%s alt=%s",
                 a, s.get("raw_score"), s.get("ref_value"), s.get("alt_value"))
log.info("wrote %s", OUT)
