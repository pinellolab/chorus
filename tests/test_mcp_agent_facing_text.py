"""What the MCP server tells an agent must be true, and checkably so.

An agent using chorus over MCP sees four things before it calls anything: the system prompt
(`_INSTRUCTIONS`), the `getting_started` prompt, `list_oracles`' specs (`ORACLE_SPECS`) and each
tool's description and parameter schema. It cannot see the README, the code or the shipped
backgrounds. So a wrong statement here is not a typo — it is the agent's model of the system.

A 2026-11-15 sweep of that text found it wrong in ways that changed what an agent would do:

* `getting_started` said "all 7 available oracles", left Cherimoya out of its menu, and put
  AlphaGenome at "5,731 tracks" — the metadata row count, of which 563 rows are `padding`
  placeholders (`tests/test_documented_track_counts.py` had fixed the same number in four docs,
  but never looked at the MCP server).
* Its example searches, 'DNASE K562' and 'CAGE liver', match **nothing**: the query is one
  substring, and the ids say 'DNASE:K562'. The truncation note advised "add an assay or cell type",
  which, done the obvious way, also returns nothing.
* `list_tracks('alphagenome', 'PROCAP')` returned 128 rows, 116 of them `PROCAP/Padding/./N`.
* `discover_variant_cell_types` described a "~638 cell type" Enformer screen that cannot run: the
  scout ids are always AlphaGenome's, so any other oracle silently returned an empty ranking.
* `load_oracle` answered `already_loaded` to a request for a *different* model and did not say
  which one it held; and `load_oracle('legnet', cell_type=...)` raised `TypeError`.

Each test below pins one of those, reading the text the way an agent receives it — through
`mcp.list_tools()` and the prompt functions, not by grepping the source.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

import pytest

pytest.importorskip("fastmcp")

from chorus.mcp import server  # noqa: E402
from chorus.mcp.server import ORACLE_SPECS, _INSTRUCTIONS, getting_started, list_tracks, mcp  # noqa: E402

#: Display names an agent needs to recognise, keyed by registry name.
DISPLAY = {
    "enformer": "Enformer",
    "borzoi": "Borzoi",
    "chrombpnet": "ChromBPNet",
    "cherimoya": "Cherimoya",
    "sei": "Sei",
    "legnet": "LegNet",
    "epinformerseq": "EPInformer-seq",
    "alphagenome": "AlphaGenome",
    "alphagenome_pt": "alphagenome_pt",
}


def _tools():
    return asyncio.run(mcp.list_tools())


def _agent_facing_text() -> dict[str, str]:
    """Every string an MCP client is shown, labelled by where it came from."""
    out = {"_INSTRUCTIONS": _INSTRUCTIONS, "getting_started": getting_started()}
    out["analyze_variant prompt"] = server.analyze_variant(
        variant="chr1:109274968 G>T", gene="SORT1", cell_type="HepG2"
    )
    for name, spec in ORACLE_SPECS.items():
        out[f"ORACLE_SPECS[{name}]"] = spec["description"]
    for tool in _tools():
        # The raw description strings, not json.dumps(parameters): dumping escapes newlines and
        # em dashes, which glues a wrapped '5,168 tracks' onto the line before ('\n5,168' reads
        # as a claim of 168) and hides an oracle name that follows one.
        params = [
            f"{pname}: {spec.get('description', '')}"
            for pname, spec in (tool.parameters or {}).get("properties", {}).items()
        ]
        out[f"tool {tool.name}"] = "\n".join([tool.description or "", *params])
    return out


def _alphagenome_queryable() -> int:
    from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata

    n = sum(1 for _ in get_metadata().iter_tracks())
    if not n:
        pytest.skip("alphagenome metadata unavailable")
    return n


#: Oracle names as prose writes them, mapped to the background key their track count is checked
#: against. alphagenome_pt borrows AlphaGenome's null, so it shares the count.
_NAME_TO_KEY = {
    "enformer": "enformer", "borzoi": "borzoi", "chrombpnet": "chrombpnet", "bpnet": "chrombpnet",
    "cherimoya": "cherimoya", "catv1": "cherimoya", "sei": "sei", "legnet": "legnet",
    "epinformer-seq": "epinformerseq", "epinformerseq": "epinformerseq",
    "alphagenome": "alphagenome", "alphagenome_pt": "alphagenome",
}
_NAME_RE = re.compile(
    r"(?<![\w-])(" + "|".join(sorted(map(re.escape, _NAME_TO_KEY), key=len, reverse=True)) + r")(?![\w-])",
    re.I,
)


def _track_count_claims(text: str):
    """Yield (oracle key, claimed count, snippet) for every 'N tracks' / 'N-track' in ``text``.

    A number is attributed to the NEAREST oracle name before it (within 160 characters), not to
    any name in the window — 'AlphaGenome ... Enformer: 5,313 tracks' is a claim about Enformer.
    """
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d{2,6})(?:-track|\s+tracks)\b", text):
        names = list(_NAME_RE.finditer(text, max(0, m.start() - 160), m.start()))
        if names:
            key = _NAME_TO_KEY[names[-1].group(1).lower()]
            yield key, int(m.group(1).replace(",", "")), text[names[-1].start(): m.end()]


def _shipped_counts() -> dict[str, int]:
    import numpy as np

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    out = {}
    if CHORUS_BACKGROUNDS_DIR.is_dir():
        for path in sorted(CHORUS_BACKGROUNDS_DIR.glob("*_pertrack.npz")):
            with np.load(path, allow_pickle=True) as data:
                out[path.name.replace("_pertrack.npz", "")] = len(data["track_ids"])
    return out


def test_registry_and_display_names_agree():
    assert set(DISPLAY) == set(ORACLE_SPECS), "update DISPLAY when an oracle is added"


def test_getting_started_names_every_oracle_with_the_system_prompts_count():
    text = getting_started()
    missing = [shown for shown in DISPLAY.values() if shown not in text]
    assert not missing, f"getting_started never mentions {missing}"

    n_oracles = len({name.removesuffix("_pt") for name in ORACLE_SPECS})
    assert f"{n_oracles} genomic deep-learning oracles" in _INSTRUCTIONS
    assert f"all {n_oracles} oracles" in text, (
        f"getting_started must state the same oracle count as the system prompt ({n_oracles})"
    )


def test_no_agent_facing_text_claims_a_stale_alphagenome_track_count():
    """5,731 is a row count; an agent reading 'tracks' takes it as what it can query.

    Runs without backgrounds (the metadata ships in the repo), so CI checks it too.
    """
    actual = _alphagenome_queryable()
    problems = [
        f"{where}: {snippet!r}"
        for where, text in _agent_facing_text().items()
        for key, claimed, snippet in _track_count_claims(text)
        if key == "alphagenome" and claimed != actual
    ]
    assert not problems, f"AlphaGenome has {actual} queryable tracks:\n" + "\n".join(problems)


def test_no_agent_facing_text_contradicts_the_shipped_backgrounds():
    """The comparison test_documented_track_counts makes for the docs, applied to the MCP text."""
    counts = _shipped_counts()
    if not counts:
        pytest.skip("no downloaded backgrounds")
    problems = [
        f"{where}: claims {claimed} for {key}, shipped {counts[key]} -- {snippet!r}"
        for where, text in _agent_facing_text().items()
        for key, claimed, snippet in _track_count_claims(text)
        if key in counts and claimed != counts[key]
    ]
    assert not problems, "\n".join(problems)


def test_the_claim_finder_catches_what_this_file_was_written_for():
    """Guard the guard: the two strings that were wrong must be flagged, and a neighbour not."""
    old = "   - **AlphaGenome** (recommended): 1Mb window, 5,731 tracks, 1bp resolution."
    assert list(_track_count_claims(old))[0][:2] == ("alphagenome", 5731)
    old_spec = "`alphagenome`, JAX). Same 5,731-track schema, same weights"
    assert list(_track_count_claims(old_spec))[0][:2] == ("alphagenome", 5731)
    neighbour = "AlphaGenome is recommended.\n   - **Enformer**: 114kb output, 5,313 tracks"
    assert list(_track_count_claims(neighbour))[0][:2] == ("enformer", 5313)


@pytest.mark.parametrize("tool_name", ["list_tracks", "load_oracle"])
def test_oracle_name_help_lists_every_registered_name(tool_name):
    tool = next(t for t in _tools() if t.name == tool_name)
    help_text = tool.parameters["properties"]["oracle_name"]["description"]
    missing = [n for n in ORACLE_SPECS if not re.search(rf"\b{n}\b", help_text)]
    assert not missing, f"{tool_name}'s oracle_name help omits {missing}"


def _quoted_examples(text: str, marker: str) -> list[str]:
    segment = text[text.index(marker):]
    segment = segment[: segment.index(")")]
    return re.findall(r"['\"]([^'\"]+)['\"]", segment)


def test_the_documented_example_searches_find_tracks():
    """An example query that matches nothing teaches the agent that the track does not exist."""
    _alphagenome_queryable()
    tool = next(t for t in _tools() if t.name == "list_tracks")
    examples = _quoted_examples(getting_started(), "relevant assays (e.g.")
    examples += _quoted_examples(tool.parameters["properties"]["query"]["description"], "(e.g.")
    assert examples, "could not find the example queries"
    for query in examples:
        assert list_tracks("alphagenome", query=query)["num_results"] > 0, (
            f"documented example query {query!r} returns no AlphaGenome tracks"
        )


#: A search that truncates on each oracle, so its reply carries the narrowing note.
_TRUNCATING = [
    ("alphagenome", "RNA"), ("enformer", "K562"), ("borzoi", "K562"),
    ("chrombpnet", "CHIP"), ("cherimoya", "ATAC"), ("sei", "K562"),
]


@pytest.mark.parametrize("oracle,query", _TRUNCATING)
def test_the_truncation_notes_example_finds_tracks_on_that_oracle(oracle, query):
    """Follow the note's advice literally: its example must return tracks on the oracle it was
    given for. One shared example cannot — 'DNASE:HepG2' finds nothing on Cherimoya or Sei."""
    result = list_tracks(oracle, query=query)
    if "error" in result:
        pytest.skip(f"{oracle} metadata unavailable: {result['error']}")
    assert result["truncated"], f"{oracle} {query!r} no longer truncates; pick a broader query"
    example = re.search(r"e\.g\. '([^']+)'", result["note"])
    assert example, f"{oracle}: the note gives no concrete query to try: {result['note']!r}"
    follow = list_tracks(oracle, query=example.group(1))
    assert follow["num_results"] > 0, f"{oracle}: the note's example {example.group(1)!r} matches nothing"


@pytest.mark.parametrize("oracle", ["enformer", "borzoi", "alphagenome", "alphagenome_pt"])
def test_list_oracles_assay_types_match_the_metadata(oracle):
    """Enformer advertised RNA; it has no RNA track."""
    if oracle == "enformer":
        from chorus.oracles.enformer_source.enformer_metadata import get_metadata
    elif oracle == "borzoi":
        from chorus.oracles.borzoi_source.borzoi_metadata import get_metadata
    else:
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
    actual = set(get_metadata().list_assay_types())
    if not actual:
        pytest.skip(f"{oracle} metadata unavailable")
    assert set(ORACLE_SPECS[oracle]["assay_types"]) == actual


# ── AlphaGenome padding rows are not tracks ───────────────────────────────

@pytest.mark.parametrize("query", ["PROCAP", "DNASE", "RNA", "Padding"])
def test_alphagenome_search_never_returns_padding(query):
    from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata

    _alphagenome_queryable()
    df = get_metadata().search_tracks(query)
    if len(df):
        assert not (df["name"].astype(str).str.lower() == "padding").any()
        assert not df["identifier"].str.contains("/Padding/").any()


def test_list_tracks_procap_is_the_twelve_real_tracks():
    _alphagenome_queryable()
    result = list_tracks("alphagenome", query="PROCAP")
    assert result["num_results"] == 12, result["num_results"]
    assert all("Padding" not in r["identifier"] for r in result["tracks"])


# ── discover_variant_cell_types is AlphaGenome-only, and says so ──────────

@pytest.mark.parametrize("oracle", ["enformer", "borzoi", "chrombpnet", "sei"])
def test_cell_type_discovery_refuses_other_oracles_loudly(oracle):
    out = server.discover_variant_cell_types(
        oracle_name=oracle, position="chr1:109274968", ref_allele="G", alt_alleles=["T"]
    )
    assert "error" in out, "must not answer with an empty ranking that reads as 'no effect'"
    assert "discover_variant" in out["error"] and "alphagenome" in out["error"]


# ── load_oracle never silently serves a different model ───────────────────

class _FakePerModelOracle:
    """Stands in for a loaded ChromBPNet: the attributes its load_pretrained_model sets, and the
    signatures the state manager introspects to route and vet selection arguments."""

    def __init__(self, assay="DNASE", cell_type="HepG2", fold=0, tf=None,
                 model_type="chrombpnet_nobias", use_environment=True, device="cpu",
                 reference_fasta=None):
        self.assay, self.cell_type, self.fold, self.tf = assay, cell_type, fold, tf
        self.model_type, self.device = model_type, device

    def load_pretrained_model(self, assay=None, cell_type=None, weights=None, TF=None, fold=0,
                              model_type="chrombpnet_nobias", is_custom=False):
        pass


@pytest.fixture
def state(monkeypatch):
    from chorus.mcp import state as state_mod

    monkeypatch.setattr(state_mod.OracleStateManager, "_instance", None)
    mgr = state_mod.OracleStateManager()
    mgr._oracles["chrombpnet"] = _FakePerModelOracle()
    mgr._load_times["chrombpnet"] = 1.0
    return mgr


def test_a_repeat_load_of_a_different_model_is_refused(state):
    with pytest.raises(ValueError, match=r"unload_oracle\('chrombpnet'\)"):
        state.load_oracle("chrombpnet", assay="DNASE", cell_type="K562")


def test_a_repeat_load_of_the_same_model_says_which_model_it_is(state):
    out = state.load_oracle("chrombpnet", assay="dnase", cell_type="HepG2")
    assert out["status"] == "already_loaded"
    assert out["loaded_model"] == {
        "assay": "DNASE", "cell_type": "HepG2", "fold": 0, "model_type": "chrombpnet_nobias",
    }
    assert "note" not in out


def test_a_bare_repeat_load_still_succeeds(state):
    assert state.load_oracle("chrombpnet")["status"] == "already_loaded"


def test_a_different_model_type_is_a_different_model(state):
    """ChromBPNet did not record model_type, so 'chrombpnet' vs 'chrombpnet_nobias' slipped by."""
    with pytest.raises(ValueError, match="model_type"):
        state.load_oracle("chrombpnet", cell_type="HepG2", model_type="chrombpnet")


def test_an_argument_the_oracle_never_takes_is_refused_by_name(state):
    """Not 'unload and load again', which would fail the same way and drop the working model."""
    with pytest.raises(ValueError, match="does not take encode_id") as err:
        state.load_oracle("chrombpnet", encode_id="ENCSR000EOT")
    assert "unload_oracle" not in str(err.value)


def test_oracle_status_reports_the_model_actually_held(state):
    (info,) = state.list_loaded()
    assert info["loaded_model"]["cell_type"] == "HepG2"


def test_a_constructor_only_selection_reaches_the_constructor(monkeypatch):
    """EPInformer-seq takes `assay` only in __init__; sent to load_pretrained_model it raised."""
    import chorus
    from chorus.mcp import state as state_mod

    built = []

    class _FakeEPI:
        def __init__(self, cell_type="K562", assay="Enhancer_DNase", use_environment=True,
                     device=None, reference_fasta=None):
            self.cell_type, self.assay, self.device = cell_type, assay, device
            built.append(assay)

        def load_pretrained_model(self, weights=None, cell_type=None):
            self.cell_type = cell_type or self.cell_type

    monkeypatch.setattr(chorus, "create_oracle", lambda name, **kw: _FakeEPI(**{
        k: v for k, v in kw.items() if k in ("cell_type", "assay", "use_environment", "device",
                                             "reference_fasta")}))
    monkeypatch.setattr(state_mod.OracleStateManager, "_instance", None)
    mgr = state_mod.OracleStateManager()
    monkeypatch.setattr(mgr, "_auto_load_normalizer", lambda name: None)
    out = mgr.load_oracle("epinformerseq", assay="Enhancer_H3K27ac", cell_type="HepG2")
    assert built[-1] == "Enhancer_H3K27ac"
    assert out["loaded_model"] == {"assay": "Enhancer_H3K27ac", "cell_type": "HepG2"}


# ── per-model oracles accept the selection arguments callers pass ─────────

#: What the MCP load path and discover_variant pass to load_pretrained_model, per oracle.
_SELECTION_KWARGS = {
    "chorus.oracles.chrombpnet:ChromBPNetOracle": {"assay", "cell_type", "TF", "fold", "model_type"},
    "chorus.oracles.cherimoya:CherimoyaOracle": {"assay", "cell_type", "fold", "encode_id"},
    "chorus.oracles.legnet:LegNetOracle": {"cell_type"},
    "chorus.oracles.epinformerseq:EPInformerSeqOracle": {"cell_type"},
}


@pytest.mark.parametrize("path,keys", sorted(_SELECTION_KWARGS.items()))
def test_load_pretrained_model_accepts_the_selection_arguments(path, keys):
    import importlib

    module, cls = path.split(":")
    oracle_cls = getattr(importlib.import_module(module), cls)
    params = inspect.signature(oracle_cls.load_pretrained_model).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    missing = keys - set(params)
    assert not missing, f"{cls}.load_pretrained_model does not accept {sorted(missing)}"


def test_legnet_switches_cell_type_at_load(monkeypatch):
    from chorus.oracles.legnet import LegNetOracle

    loads = []
    monkeypatch.setattr(LegNetOracle, "_check_env_ready", lambda self: None)
    monkeypatch.setattr(LegNetOracle, "_load_in_environment", lambda self: loads.append(self.cell_type))
    oracle = LegNetOracle(use_environment=True)
    oracle.use_environment = True
    oracle.load_pretrained_model(cell_type="K562")
    assert (oracle.cell_type, oracle.assay_id, loads) == ("K562", "LentiMPRA:K562", ["K562"])
    with pytest.raises(Exception, match="not in available cell types"):
        oracle.load_pretrained_model(cell_type="nonexistent")
