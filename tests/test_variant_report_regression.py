"""Regression guards for chorus.analysis.variant_report."""
import inspect

from chorus.analysis import variant_report


def test_build_variant_report_no_local_layer_configs_shadow():
    """`build_variant_report` must NOT re-import LAYER_CONFIGS function-locally.

    LAYER_CONFIGS is imported at module scope. A function-local
    `from .scorers import LAYER_CONFIGS` makes Python treat the name as a local
    for the whole function, so its *earlier* use (the per-gene TSS block) raises
    UnboundLocalError — which crashed fine_map_causal_variant /
    analyze_variant_multilayer on any CAGE track with a nearby gene.
    """
    src = inspect.getsource(variant_report.build_variant_report)
    assert "import LAYER_CONFIGS" not in src, (
        "function-local LAYER_CONFIGS import shadows the module-level one "
        "(UnboundLocalError on the CAGE + nearby-gene path)"
    )
