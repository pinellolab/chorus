"""``discover_variant`` used to render and write its HTML report twice.

``chorus/mcp/server.py::discover_variant`` called ``discover_variant_effects(...,
output_path=state.output_dir)``, which builds the ``VariantReport`` and writes its HTML
right there -- but without an ``analysis_request``, since the server only built one
*after* that call returned. The server then stamped ``report.analysis_request`` and
called ``_write_html_report`` a second time to get the user prompt into the file. Same
report, same output file, rendered and written twice -- expensive when
``show_conservation`` makes each render nontrivial, and wasted even when it doesn't.

``discover_variant_effects`` already accepts an ``analysis_request`` kwarg for exactly
this ("stamped onto the report so the HTML is rendered with the user prompt on first
write (avoids a post-hoc ``to_html`` rewrite)") -- the server just wasn't using it. The
fix builds the ``AnalysisRequest`` up front and passes it in, so the report renders
correctly on its one and only write; the server now looks up the resulting path instead
of re-rendering it.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _unwrap(fn):
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)
    return fn


@pytest.fixture
def fake_report(tmp_path):
    """Stand-in for VariantReport: records to_html calls and where they'd land."""
    report = MagicMock()
    report.to_markdown.return_value = "# markdown"
    report.resolve_html_path.side_effect = lambda output_dir: tmp_path / "report.html"
    return report


def test_discover_variant_renders_and_writes_exactly_once(monkeypatch, fake_report, tmp_path):
    import chorus.mcp.server as server

    fake_state = SimpleNamespace(
        get_oracle=lambda name: object(),
        get_normalizer=lambda name: None,
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(server, "_state", lambda: fake_state)

    captured_kwargs = {}

    def fake_discover_variant_effects(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"report": fake_report, "layer_rankings": {}, "cell_type_ranking": []}

    monkeypatch.setattr(
        "chorus.analysis.discovery.discover_variant_effects", fake_discover_variant_effects
    )

    discover_variant = _unwrap(server.discover_variant)
    result = discover_variant(
        oracle_name="enformer",
        position="chr1:109274968",
        ref_allele="G",
        alt_alleles=["T"],
        user_prompt="what does this variant do",
    )

    # The report must render with the user prompt already attached -- not stamped on
    # after discover_variant_effects has already written it once.
    assert captured_kwargs["analysis_request"] is not None
    assert captured_kwargs["analysis_request"].user_prompt == "what does this variant do"
    assert captured_kwargs["analysis_request"].tool_name == "discover_variant"

    # No second render/write: the report's own to_html must never be called from the
    # server (discover_variant_effects, which we've stubbed out, is the only writer).
    fake_report.to_html.assert_not_called()

    # The path is looked up, not re-derived by writing again.
    fake_report.resolve_html_path.assert_called_once_with(str(tmp_path))
    assert result["html_report_path"] == str(tmp_path / "report.html")
    assert result["markdown_report"] == "# markdown"
