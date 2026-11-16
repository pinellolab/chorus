"""A walkthrough README may not quote a number its own artefact disagrees with.

The regeneration scripts rewrite ``example_output.{json,md,tsv}`` and the HTML. They
have never touched ``README.md``, which is hand-written prose — so every correctness
fix that moved a number left the narrative behind. A consistency audit found 13
READMEs carrying 63 numeric claims, and the divergence was not marginal:

* ``variant_analysis/SORT1_chrombpnet/README.md`` describes an **ATAC** run at
  **-0.111** ("moderate closing") and builds a whole section on why AlphaGenome and
  ChromBPNet *disagree*. The committed artefact is **DNASE** at **+1.376** — same
  direction, strong opening. The section explains a contradiction that no longer
  exists.
* ``variant_analysis/SORT1_enformer/README.md`` lists six top tracks, five of which
  appear nowhere in the artefact.
* ``discovery/README.md`` names three top cell types that appear nowhere in the
  committed discovery output.
* Several quote pre-#92 effect sizes that are 7-11x smaller than the current ones.

The numbers moved for good reasons — the #149 denominator fix alone corrected the RNA
numerator by 251-1736x — but a README that states the old ones is simply wrong, and a
reader has no way to tell which of the two to believe.

This test does not check prose. It extracts every signed decimal that looks like an
effect size and requires it to appear somewhere in the sibling artefact. That is
deliberately narrow: it catches the stale-number class without trying to referee
wording, and it fails loudly when a fix moves a number so the README is updated in
the same commit.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

WALKTHROUGHS = Path(__file__).resolve().parent.parent / "examples" / "walkthroughs"

# A signed decimal with at least two fraction digits: +1.376, -0.111, +3.316.
# Two digits minimum keeps out version strings, section numbers and p-values.
#
# The sign class must include U+2212 MINUS SIGN, not just ASCII hyphen-minus. Markdown
# written by a human -- or pasted from a rendered table -- routinely uses the typographic
# minus, and an ASCII-only class silently sees NOTHING: this test reported
# SKIPPED "no effect-size claims" for region_swap/README.md, a file that is *entirely* a
# table of five effect claims, all of them stale, all written with U+2212. A guard that
# skips a file is indistinguishable from a guard that passes it.
_MINUS = "\u2212"
_EFFECT = re.compile(rf"(?<![\w.])([+\-{_MINUS}]\d+\.\d{{2,}})(?![\w])")


def _normalise_sign(claim: str) -> str:
    """U+2212 -> ASCII '-' so float() accepts it."""
    return claim.replace(_MINUS, "-")

# Numbers that appear in prose for reasons other than quoting this artefact.
_ALLOWED = {
    # Thresholds and worked arithmetic, not measurements.
    "+0.00", "-0.00",
}
_TOLERANCE = 5e-3      # READMEs round; the artefact is full precision.


def _artefact_numbers(directory: Path) -> set[float]:
    """Every numeric leaf in this walkthrough's JSON, plus its TSV cells."""
    out: set[float] = set()
    js = directory / "example_output.json"
    if js.exists():
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif isinstance(o, bool) or o is None:
                return
            elif isinstance(o, (int, float)):
                out.add(float(o))
        walk(json.loads(js.read_text()))
    tsv = directory / "example_output.tsv"
    if tsv.exists():
        for tok in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", tsv.read_text()):
            try:
                out.add(float(tok))
            except ValueError:
                pass
    return out


# --- Table-style walkthroughs ---------------------------------------------------------------
#
# ``tissue_ranking/`` has no ``example_output.json``: its artefact is a set of CSV tables and one
# markdown report in numbered folders. Reading it like the others found nothing of its own, fell
# back to the sibling it links to, and so checked its README against the wrong walkthrough.
#
# Pooling every number in those tables is not an answer either: they hold ~49,000 values, so a
# +-5e-3 match would pass almost any three-decimal claim. A table-style walkthrough is therefore
# checked against a SCOPED pool (the markdown report, the per-gene table, and the raw scores of
# the cell type it follows) and a claim must be that value ROUNDED to the claim's own decimals.
# What this still cannot see is a stale claim that happens to equal another value in the pool;
# the row-aware checks in ``test_tissue_ranking_readme_tables_match_their_csvs`` carry the rest.

def _is_table_walkthrough(directory: Path) -> bool:
    return (not any(directory.rglob("example_output.json"))
            and any(directory.rglob("*_track_scores.csv")))


def _read_csv(path: Path) -> list[dict]:
    """Rows of a discovery / ranking table, skipping its ``#`` header comments."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def _target_cell_type(directory: Path) -> str | None:
    """The cell type a table-style walkthrough follows, from its ``NN_multilayer_<cell type>`` folder."""
    for p in sorted(directory.glob("*_multilayer_*")):
        m = re.match(r"\d+_multilayer_(.+)$", p.name)
        if m and p.is_dir():
            return m.group(1)
    return None


def _table_pool(directory: Path) -> list[float]:
    """The numbers a table-style README may quote: its reports, its gene table, the target's raw scores."""
    pool: list[float] = []
    number = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
    for md in directory.rglob("*.md"):
        if md.name not in ("README.md", "prompt.md"):
            pool += [float(t) for t in number.findall(md.read_text())]
    for p in directory.rglob("gene_response_by_gene.csv"):
        for row in _read_csv(p):
            for v in row.values():
                if number.fullmatch(v or ""):
                    pool.append(float(v))
    target = (_target_cell_type(directory) or "").lower()
    if target:
        for p in directory.rglob("*_track_scores.csv"):
            pool += [float(r["raw_score"]) for r in _read_csv(p) if r["cell_type_key"].lower() == target]
        for p in directory.rglob("gene_response_by_track.csv"):
            pool += [float(r["raw_score"]) for r in _read_csv(p)
                     if r["cell_type"].lower() == target and r["raw_score"]]
    return pool


def _in_pool(claim: str, pool: list[float]) -> bool:
    """True if some pooled value, rounded to the claim's own number of decimals, IS the claim."""
    want = float(_normalise_sign(claim))
    decimals = len(claim.split(".")[1])
    return any(round(v, decimals) == want for v in pool)


def _cases():
    if not WALKTHROUGHS.is_dir():
        return []
    out = []
    for readme in sorted(WALKTHROUGHS.rglob("README.md")):
        # Only leaf walkthroughs have their own artefact; category-level READMEs
        # (variant_analysis/README.md and friends) summarise several and are checked
        # against the union of their children.
        out.append(pytest.param(
            readme, id=str(readme.parent.relative_to(WALKTHROUGHS)) or "root"))
    return out


@pytest.mark.parametrize("readme", _cases())
def test_readme_effect_sizes_appear_in_the_artefacts(readme: Path):
    claims = [c for c in _EFFECT.findall(readme.read_text())
              if _normalise_sign(c)[:5] not in _ALLOWED]
    if not claims:
        pytest.skip("no effect-size claims")

    text = readme.read_text()

    if _is_table_walkthrough(readme.parent):
        # Its artefact is its own tables, and nothing a linked sibling holds may excuse a
        # claim: a table-style README is checked against a scoped pool, rounded exactly.
        pool = _table_pool(readme.parent)
        if not pool:
            pytest.skip("its tables carry no numbers")

        def _present(claim: str) -> bool:
            return _in_pool(claim, pool)
    else:
        # A leaf README is checked against its own artefact; a category README against
        # the union over its subdirectories, since it summarises all of them.
        dirs = [readme.parent]
        if not (readme.parent / "example_output.json").exists():
            dirs = [p.parent for p in readme.parent.rglob("example_output.json")]

        # ... PLUS any walkthrough this one links to. A "Cross-oracle comparison" section
        # legitimately quotes a SIBLING's numbers ("compare with the ChromBPNet analysis
        # (+1.376 DNASE:HepG2)"), and those live in the sibling's artefact, not this one.
        # Following the link is the point: it makes a cross-reference checkable rather
        # than exempt, so a stale one still fails. SORT1_enformer had asserted
        # "-0.111 ATAC:HepG2" about a sibling whose own README said that figure was stale
        # prose -- two files disagreeing, with nothing checking either.
        for m in re.finditer(r"\]\(((?:\.\./)+[A-Za-z0-9_./-]+)\)", text):
            target = (readme.parent / m.group(1)).resolve()
            if not target.is_dir():
                target = target.parent
            try:
                target.relative_to(WALKTHROUGHS.resolve())
            except ValueError:
                continue  # points outside the walkthroughs tree
            dirs += [q.parent for q in target.rglob("example_output.json")]

        dirs = list(dict.fromkeys(dirs))
        if not dirs:
            pytest.skip("no artefact to check against")

        available: set[float] = set()
        for d in dirs:
            available |= _artefact_numbers(d)
        if not available:
            pytest.skip("artefacts carry no numbers")

        def _present(claim: str) -> bool:
            want = float(_normalise_sign(claim))
            return any(abs(want - got) <= _TOLERANCE for got in available)

    # A line that explicitly documents a number as HISTORICAL is doing the right
    # thing, and rewriting it would erase the correction. Recognised narrowly:
    # the line must say so in words, so it cannot be used to park a live claim.
    _HISTORY = re.compile(
        r"earlier version|earlier revision|used to|previously|was wrong|were wrong"
        r"|no longer|stale prose|audit's recorded|before this|superseded",
        re.I,
    )

    def _is_history(claim: str) -> bool:
        """True only if EVERY occurrence is marked historical.

        Two markers count. A line saying so in words, and an ``old -> new`` arrow,
        where the value on the left of the arrow is by construction the superseded
        one -- "composite 0.995 -> 0.999, alt effect -1.363 -> -1.368" documents a
        change, and demanding the left side exist in the current artefact would make
        it impossible to record one.
        """
        for m in re.finditer(re.escape(claim), text):
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            line = text[ls:le if le > 0 else len(text)]
            after = text[m.end():m.end() + 12]
            if _HISTORY.search(line):
                continue
            if re.match(r"\s*(\u2192|->|=>)", after):   # left side of old -> new
                continue
            return False   # at least one LIVE use of this number
        return True

    stale = []
    for claim in claims:
        if _present(claim):
            continue
        if _is_history(claim):
            continue
        stale.append(claim)

    assert not stale, (
        f"{readme.relative_to(WALKTHROUGHS.parent.parent)} quotes "
        f"{len(stale)} effect size(s) absent from its artefact(s): "
        f"{stale[:8]}. The regeneration scripts rewrite the JSON/MD/TSV/HTML but "
        f"never the README, so a correctness fix leaves the prose behind. Update "
        f"the README in the same commit as the number."
    )


# --- Row-aware checks for a table-style README -----------------------------------------------
#
# The effect-size check above cannot see an INTEGER, and the integers are what went stale when the
# background nulls were refreshed: tissue ranks, "11 of 37", "12 / 12". It also cannot tell a
# stale effect from another value that happens to round the same. These compare each README table
# row, and each sentence that quotes a rank or a count, with the CSV or report it is read from.

_LAYER_COLUMN = {"DNase": "DNase", "ATAC": "ATAC", "ChIP-TF": "ChIP-TF", "Histone": "ChIP-histone",
                 "histone": "ChIP-histone", "RNA-seq": "RNA-seq", "CAGE": "CAGE"}


def _ranking_walkthroughs():
    if not WALKTHROUGHS.is_dir():
        return []
    return [pytest.param(p.parent.parent, id=str(p.parent.parent.relative_to(WALKTHROUGHS)))
            for p in sorted(WALKTHROUGHS.rglob("02_ranking_tissue/layer_rank_table.csv"))]


def _md_table(text: str, header_prefix: str) -> tuple[list[str], list[list[str]]]:
    """``(header cells, body rows)`` of the markdown table whose header line starts with *header_prefix*."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(header_prefix)), None)
    assert start is not None, f"no table starting {header_prefix!r}"

    def cells(ln):
        return [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]

    rows = []
    for ln in lines[start + 2:]:          # skip the header and its |---| rule
        if not ln.startswith("|"):
            break
        rows.append(cells(ln))
    return cells(lines[start]), rows


def _rank(cell) -> int | None:
    """A rank cell: blank or a dash means the tissue has no track in that layer."""
    cell = (cell or "").strip()
    return None if cell in ("", "–", "-", "nan", "NA") else int(float(cell))


@pytest.mark.parametrize("directory", _ranking_walkthroughs())
def test_tissue_ranking_readme_tables_match_their_csvs(directory: Path):
    text = (directory / "README.md").read_text()
    problems: list[str] = []
    tissue = {r["group"]: r for r in _read_csv(directory / "02_ranking_tissue" / "layer_rank_table.csv")}
    system = {r["group"]: r for r in _read_csv(directory / "02_ranking_system" / "layer_rank_table.csv")}

    def by_global(table):
        """Group names, best first; a group with no Global rank sorts last."""
        return sorted(table, key=lambda g: (_rank(table[g]["Global"]) is None,
                                            _rank(table[g]["Global"]) or 0))

    # 1. The tissue table: every row, every layer, and that the rows ARE the top of the CSV.
    header, rows = _md_table(text, "| Global | Tissue |")
    for cells in rows:
        rec = dict(zip(header, cells))
        row = tissue.get(rec["Tissue"])
        if row is None:
            problems.append(f"tissue table: {rec['Tissue']!r} is not in 02_ranking_tissue/layer_rank_table.csv")
            continue
        for label, col in [("Global", "Global")] + [(k, v) for k, v in _LAYER_COLUMN.items() if k in rec]:
            if _rank(rec[label]) != _rank(row[col]):
                problems.append(f"tissue table {rec['Tissue']} {label}: README {rec[label]!r}, csv {row[col]!r}")
    if [r[1] for r in rows] != by_global(tissue)[:len(rows)]:
        problems.append(f"tissue table lists {[r[1] for r in rows]}, but the csv's top {len(rows)} by Global "
                        f"are {by_global(tissue)[:len(rows)]}")

    # 2. The system sentence: "System level: **Liver** (1), Adipose (2), ..."
    m = re.search(r"System level: (.+?)\.\n", text)
    named = [(n.strip(), int(k)) for n, k in
             re.findall(r"([^,()]+?) \((\d+)\)", m.group(1).replace("**", ""))] if m else []
    if not named:
        problems.append("no 'System level: ... (1), ... (2)' sentence found")
    for name, k in named:
        if name not in system or _rank(system[name]["Global"]) != k:
            problems.append(f"system level: README puts {name!r} at {k}, csv has "
                            f"{system.get(name, {}).get('Global')!r}")
    if [n for n, _ in named] != by_global(system)[:len(named)]:
        problems.append(f"system level lists {[n for n, _ in named]}, csv top {len(named)} by Global "
                        f"are {by_global(system)[:len(named)]}")

    # 3. "Liver's per-layer ranks there are DNase 6, ATAC 4, ChIP-TF 1, ..." (the system csv)
    m = re.search(r"(\w[\w /]*)'s per-layer ranks there are (.+?)\.\n", text)
    if not m or m.group(1).strip() not in system:
        problems.append("no \"<group>'s per-layer ranks there are ...\" sentence for a system in the csv")
    else:
        for label, k in re.findall(r"([\w-]+) (\d+)", m.group(2)):
            if _rank(system[m.group(1).strip()][_LAYER_COLUMN[label]]) != int(k):
                problems.append(f"per-layer ranks, {m.group(1).strip()} {label}: README {k}, csv "
                                f"{system[m.group(1).strip()][_LAYER_COLUMN[label]]!r}")

    # 4. "chromatin ranks are only 11 of 37 (DNase) and 9 of 33 (ATAC)": the top tissue's rank and the
    #    number of tissues ranked in that layer.
    summary = json.loads((directory / "02_ranking_tissue" / "run_summary.json").read_text())
    top = tissue[by_global(tissue)[0]]
    for k, n, label in re.findall(r"(\d+) of (\d+)\s+\((DNase|ATAC)\)", text):
        if _rank(top[label]) != int(k) or summary["layers"][label]["n_groups_ranked"] != int(n):
            problems.append(f"{label}: README {k} of {n}, csv {top[label]} of "
                            f"{summary['layers'][label]['n_groups_ranked']}")

    # 5. The gene table.
    genes = {r["gene"]: r for r in _read_csv(next(directory.glob("03_gene_response_*/gene_response_by_gene.csv")))}
    header, rows = _md_table(text, "| Gene | Median log2FC")
    for gene, median, outlier, above in (r[:4] for r in rows):
        row = genes.get(gene)
        if row is None:
            problems.append(f"gene table: {gene!r} is not in gene_response_by_gene.csv")
            continue
        decimals = len(median.split(".")[1])
        if round(float(row["median_log2fc"]), decimals) != float(median):
            problems.append(f"gene table {gene} median: README {median}, csv {row['median_log2fc']}")
        for label, got, key in (("outlier", outlier, "n_outlier"), ("above floor", above, "n_above_floor")):
            if got != f"{row[key]} / {row['n_tracks']}":
                problems.append(f"gene table {gene} {label}: README {got!r}, csv {row[key]} / {row['n_tracks']}")

    # 6. The four-layers table: every effect must sit in the same report row as one of its percentiles.
    report = next(directory.glob("04_multilayer_*/multilayer_report.md")).read_text()
    report_rows = [ln for ln in report.splitlines() if ln.startswith("|")]
    header, rows = _md_table(text, "| Layer | Track |")
    for cells in rows:
        rec = dict(zip(header, cells))
        pcts = re.findall(r"\d\.\d+", rec["Effect %ile"])
        for effect in re.findall(r"[+\-−]\d+\.\d+", rec["Effect (log2FC)"]):
            e = _normalise_sign(effect)
            if not any(f"| {e} |" in ln and any(f"| {p} |" in ln for p in pcts) for ln in report_rows):
                problems.append(f"layers table {rec['Track']}: effect {effect} with percentile {pcts} is not "
                                f"one row of the multilayer report")

    # 7. The discovery counts.
    def n_rows(oracle, layer=None):
        rows = _read_csv(next(directory.glob(f"01_discovery/*_{oracle}_track_scores.csv")))
        return len([r for r in rows if layer is None or r["molecular_layer"] == layer])

    def stated(pattern):
        m = re.search(pattern, text)
        return [int(g.replace(",", "")) for g in m.groups()] if m else None

    n_genes = len(_read_csv(next(directory.glob("01_discovery/*_alphagenome_rna_gene_percentiles.csv"))))
    for what, got, want in (
        ("AlphaGenome tracks / RNA genes", stated(r"\| AlphaGenome \| ([\d,]+) tracks; (\d+) genes"),
         [n_rows("alphagenome"), n_genes]),
        ("Cherimoya tracks / ATAC / DNase", stated(r"([\d,]+) tracks \((\d+) ATAC, (\d+) DNase\)"),
         [n_rows("cherimoya"), n_rows("cherimoya", "ATAC"), n_rows("cherimoya", "DNase")]),
        ("ChromBPNet models / ChIP-TF / DNase / ATAC",
         stated(r"([\d,]+) models: (\d+) ChIP-TF \+ (\d+) DNase \+ (\d+) ATAC"),
         [n_rows("chrombpnet"), n_rows("chrombpnet", "ChIP-TF"), n_rows("chrombpnet", "DNase"),
          n_rows("chrombpnet", "ATAC")]),
    ):
        if got != want:
            problems.append(f"discovery counts, {what}: README {got}, tables {want}")

    assert not problems, (
        f"{directory.relative_to(WALKTHROUGHS)}/README.md disagrees with its own tables in "
        f"{len(problems)} place(s):\n  " + "\n  ".join(problems[:20]))
