"""Linkage disequilibrium variant lookup.

Fetches LD proxy variants from the LDlink REST API, or converts
user-provided variant lists into a standard format.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LDLINK_CONFIG_PATH = Path.home() / ".chorus" / "config.toml"


def _resolve_ldlink_token(explicit: Optional[str]) -> Optional[str]:
    """Resolve an LDlink token via: arg -> env -> ~/.chorus/config.toml."""
    if explicit:
        return explicit
    env = os.environ.get("LDLINK_TOKEN")
    if env:
        return env
    if not _LDLINK_CONFIG_PATH.exists():
        return None
    try:
        import tomllib  # py311+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            return None
    try:
        data = tomllib.loads(_LDLINK_CONFIG_PATH.read_text())
    except Exception:
        return None
    tokens = data.get("tokens", {}) if isinstance(data, dict) else {}
    return tokens.get("ldlink") if isinstance(tokens, dict) else None


class LDLinkError(Exception):
    """Raised when LDlink API is unavailable or returns an error."""


@dataclass
class LDVariant:
    """A variant in linkage disequilibrium with a sentinel.

    ``ref``/``alt`` are stored after normalization: LDlink ``"-"`` is
    converted to ``""`` (empty string) for insertions/deletions. The
    ``kind`` field classifies the variant for downstream filtering
    (e.g. the ``snvs_only`` switch on ``fetch_ld_variants`` /
    ``prioritize_causal_variants``).
    """

    variant_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    r2: float
    dprime: float = 1.0
    distance: int = 0
    is_sentinel: bool = False
    kind: str = "snv"  # "snv" | "insertion" | "deletion" | "mnv" | "complex"


_GENOME_BUILD_ALIASES = {
    "hg38": "grch38",
    "GRCh38": "grch38",
    "grch38": "grch38",
    "hg19": "grch37",
    "GRCh37": "grch37",
    "grch37": "grch37",
}


def fetch_ld_variants(
    variant_id: str,
    population: str = "CEU",
    r2_threshold: float = 0.8,
    token: str | None = None,
    timeout: float = 30.0,
    genome_build: str = "grch38",
    snvs_only: bool = False,
) -> list[LDVariant]:
    """Query LDlink LDproxy API and return LD variants above r2 threshold.

    Args:
        variant_id: rsID (e.g. "rs12740374") or "chr1:109274968".
        population: 1000 Genomes population code (default "CEU").
        r2_threshold: Minimum r² to include (default 0.8).
        token: LDlink API token. Register free at
            https://ldlink.nih.gov/?tab=apiaccess
        timeout: Request timeout in seconds.
        genome_build: Reference build for the LDlink query. Accepts
            ``"grch38"`` / ``"hg38"`` (default) or ``"grch37"`` / ``"hg19"``.
        snvs_only: When True, drop any returned LDVariant whose
            ``kind`` is not ``"snv"`` (i.e. exclude insertions,
            deletions, MNVs, and complex multi-base changes). Default
            False — score every proxy regardless of variant type.

    Returns:
        List of LDVariant objects, sentinel first.

    Raises:
        LDLinkError: If the API is unavailable or token is missing.
        ValueError: If ``genome_build`` isn't recognised.
    """
    token = _resolve_ldlink_token(token)
    if token is None:
        raise LDLinkError(
            "LDlink API token required. Register free at "
            "https://ldlink.nih.gov/?tab=apiaccess, then either: "
            "(a) pass ldlink_token=..., (b) set LDLINK_TOKEN, or "
            "(c) run 'chorus setup all' to be prompted once."
        )

    if genome_build not in _GENOME_BUILD_ALIASES:
        raise ValueError(
            f"Unknown genome_build {genome_build!r}. "
            f"Choose one of {sorted(set(_GENOME_BUILD_ALIASES))}."
        )
    resolved_build = _GENOME_BUILD_ALIASES[genome_build]

    import requests

    url = "https://ldlink.nih.gov/LDlinkRest/ldproxy"
    params = {
        "var": variant_id,
        "pop": population,
        "r2_d": "r2",
        "token": token,
        "genome_build": resolved_build,
    }

    logger.info("Querying LDlink LDproxy for %s in %s...", variant_id, population)

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LDLinkError(f"LDlink API request failed: {exc}") from exc

    text = resp.text
    if "error" in text.lower() and len(text) < 500:
        raise LDLinkError(f"LDlink API error: {text.strip()}")

    variants = parse_ld_response(text, r2_threshold=r2_threshold)
    if snvs_only:
        n_before = len(variants)
        variants = [v for v in variants if v.kind == "snv"]
        n_dropped = n_before - len(variants)
        if n_dropped:
            logger.info("snvs_only filter dropped %d non-SNV proxies", n_dropped)
    logger.info("Found %d variants in LD (r² >= %.2f)", len(variants), r2_threshold)
    return variants


def _extract_allele_pairs(
    alleles_field: str,
    correlated_field: str | None,
) -> list[tuple[str, str]]:
    """Return a list of ``(ref, alt)`` pairs for an LDlink row.

    Three sources are tried in order:

    1. ``Correlated_Alleles`` of shape ``"SENT=PROXY,SENT=PROXY"``
       (the LDlink LDproxy column showing sentinel↔proxy allele
       co-segregation). When both pairs have non-empty alleles after
       normalisation, emit **two** ``(SENT, PROXY)`` records so the
       prioritization function can score each haplotype-specific
       transition independently.
    2. ``Alleles`` of shape ``"(REF/ALT1,ALT2,...)"`` — comma-split the
       alt half and emit one record per alt.
    3. ``Alleles`` of shape ``"(REF/ALT)"`` — a single record.

    Empty / ``"-"`` alleles are normalised via
    :func:`chorus.utils.sequence.normalize_allele` (LDlink uses ``"-"``
    for indels; chorus internally uses ``""``). Rows that fail to
    produce any valid (ref, alt) pair return an empty list and the
    caller should skip them.
    """
    from .sequence import normalize_allele
    from ..core.exceptions import InvalidRegionError

    # 1) Correlated_Alleles fanout
    if correlated_field and correlated_field.strip() not in ("", "NA", "."):
        pairs_out: list[tuple[str, str]] = []
        for entry in correlated_field.split(","):
            entry = entry.strip()
            if "=" not in entry:
                pairs_out = []
                break
            sent_raw, prox_raw = entry.split("=", 1)
            try:
                sent = normalize_allele(sent_raw)
                prox = normalize_allele(prox_raw)
            except InvalidRegionError:
                pairs_out = []
                break
            # Both ref (sentinel allele) and alt (proxy allele) being
            # empty would mean "no change" — skip.
            if sent == "" and prox == "":
                continue
            pairs_out.append((sent, prox))
        if pairs_out:
            return pairs_out

    # 2/3) Alleles column (with comma-split fallback for multi-allelic alt)
    allele_str = alleles_field.strip().strip("()")
    if "/" not in allele_str:
        return []
    ref_raw, alt_raw = allele_str.split("/", 1)
    try:
        ref = normalize_allele(ref_raw)
    except InvalidRegionError:
        return []

    pairs_out = []
    for alt_raw_one in alt_raw.split(","):
        try:
            alt = normalize_allele(alt_raw_one)
        except InvalidRegionError:
            continue
        if ref == "" and alt == "":
            continue
        pairs_out.append((ref, alt))
    return pairs_out


def parse_ld_response(
    text: str,
    r2_threshold: float = 0.8,
) -> list[LDVariant]:
    """Parse tab-separated LDlink LDproxy response text.

    Indels are returned with empty-string alleles (LDlink's ``"-"`` is
    normalised). Multi-allelic rows fan out into multiple
    :class:`LDVariant` entries — see :func:`_extract_allele_pairs`.

    Args:
        text: Raw TSV response from LDproxy API.
        r2_threshold: Minimum r² to include.

    Returns:
        List of LDVariant objects, sentinel first.
    """
    from .sequence import classify_variant

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    # Find column indices
    col_map = {col.strip(): i for i, col in enumerate(header)}

    # Expected columns: RS_Number, Coord, Alleles, MAF, Distance, Dprime, R2, ...
    rs_col = col_map.get("RS_Number", col_map.get("rs_number", 0))
    coord_col = col_map.get("Coord", col_map.get("coord", 1))
    alleles_col = col_map.get("Alleles", col_map.get("alleles", 2))
    dist_col = col_map.get("Distance", col_map.get("distance", 4))
    dprime_col = col_map.get("Dprime", col_map.get("dprime", 5))
    r2_col = col_map.get("R2", col_map.get("r2", 6))
    correlated_col = col_map.get(
        "Correlated_Alleles",
        col_map.get("correlated_alleles", None),
    )

    variants: list[LDVariant] = []

    for i, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) < max(rs_col, coord_col, alleles_col, r2_col) + 1:
            continue

        try:
            r2_val = float(fields[r2_col])
        except (ValueError, IndexError):
            continue

        if r2_val < r2_threshold and i > 0:
            continue

        # Parse coordinate: "chr1:109274968"
        coord = fields[coord_col].strip()
        if ":" not in coord:
            continue
        chrom, pos_str = coord.split(":", 1)
        try:
            position = int(pos_str)
        except ValueError:
            continue

        # Ensure chr prefix
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        correlated_raw = (
            fields[correlated_col].strip()
            if correlated_col is not None and correlated_col < len(fields)
            else None
        )
        allele_pairs = _extract_allele_pairs(
            fields[alleles_col], correlated_raw,
        )
        if not allele_pairs:
            continue

        # Parse other fields
        rs_id = fields[rs_col].strip()
        try:
            dprime = float(fields[dprime_col])
        except (ValueError, IndexError):
            dprime = 1.0
        try:
            distance = int(fields[dist_col])
        except (ValueError, IndexError):
            distance = 0

        for ref, alt in allele_pairs:
            variants.append(LDVariant(
                variant_id=rs_id if rs_id and rs_id != "." else f"{chrom}:{position}",
                chrom=chrom,
                position=position,
                ref=ref,
                alt=alt,
                r2=r2_val,
                dprime=dprime,
                distance=distance,
                is_sentinel=(i == 0),
                kind=classify_variant(ref, alt),
            ))

    return variants


def ld_variants_from_list(
    variants: list[dict],
    sentinel_id: str | None = None,
) -> list[LDVariant]:
    """Convert user-provided variant dicts to LDVariant objects.

    Args:
        variants: List of dicts with keys: chrom, pos, ref, alt.
            Optional keys: id, r2 (default 1.0).
        sentinel_id: Variant ID to mark as sentinel. If None, the
            first variant is treated as sentinel.

    Returns:
        List of LDVariant objects.
    """
    result: list[LDVariant] = []
    for i, v in enumerate(variants):
        vid = v.get("id", f"{v['chrom']}:{v['pos']}")
        is_sent = (vid == sentinel_id) if sentinel_id else (i == 0)
        result.append(LDVariant(
            variant_id=vid,
            chrom=v["chrom"],
            position=int(v["pos"]),
            ref=v["ref"],
            alt=v["alt"],
            r2=float(v.get("r2", 1.0)),
            is_sentinel=is_sent,
        ))
    return result
