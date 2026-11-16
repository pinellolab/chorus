"""An rsID passed as ``position`` must resolve to coordinates, not crash.

``analyze_variant_multilayer``, ``discover_variant`` and ``discover_variant_cell_types`` accept
``position="rs12740374"`` and resolve it through ``_resolve_rsid_to_position``, which reads the
sentinel row LDlink returns. That function read ``v.pos``, but ``LDVariant`` names the field
``position``. So every *successful* lookup raised ``AttributeError``, and the feature advertised in
all three tool descriptions had never worked since it was added (#84). Only the failure paths, an
LDlink error or an unknown rsID, were ever exercised, which is why nothing noticed.

Found by the v0.8.0 release audit's LDlink-redaction work (2026-11-16). Offline: ``fetch_ld_variants``
is replaced by a stub returning the shape LDlink's parser produces.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")

from chorus.utils.ld import LDVariant  # noqa: E402


def _sentinel(**overrides) -> LDVariant:
    fields = dict(variant_id="rs12740374", chrom="chr1", position=109274968, ref="G", alt="T",
                  r2=1.0, is_sentinel=True)
    fields.update(overrides)
    return LDVariant(**fields)


@pytest.fixture
def ldlink_returns(monkeypatch):
    def install(rows):
        import chorus.utils.ld as ld

        monkeypatch.setattr(ld, "fetch_ld_variants", lambda *a, **k: rows)
    return install


def test_a_resolved_rsid_gives_coordinates_and_alleles(ldlink_returns):
    from chorus.mcp.server import _resolve_rsid_to_position

    ldlink_returns([_sentinel()])
    assert _resolve_rsid_to_position("rs12740374", ldlink_token="unused") == (
        "chr1:109274968", "G", "T",
    )


def test_a_record_without_alleles_names_the_position_it_found(ldlink_returns):
    """The error branch read the same missing attribute, so it crashed instead of explaining."""
    from chorus.mcp.server import _resolve_rsid_to_position

    ldlink_returns([_sentinel(ref="", alt="")])
    with pytest.raises(ValueError, match=r"chr1:109274968 but no alleles"):
        _resolve_rsid_to_position("rs12740374", ldlink_token="unused")
