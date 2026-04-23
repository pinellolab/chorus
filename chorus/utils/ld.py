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
    """A variant in linkage disequilibrium with a sentinel."""

    variant_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    r2: float
    dprime: float = 1.0
    distance: int = 0
    is_sentinel: bool = False


def fetch_ld_variants(
    variant_id: str,
    population: str = "CEU",
    r2_threshold: float = 0.8,
    token: str | None = None,
    timeout: float = 30.0,
) -> list[LDVariant]:
    """Query LDlink LDproxy API and return LD variants above r2 threshold.

    Args:
        variant_id: rsID (e.g. "rs12740374") or "chr1:109274968".
        population: 1000 Genomes population code (default "CEU").
        r2_threshold: Minimum r² to include (default 0.8).
        token: LDlink API token. Register free at
            https://ldlink.nih.gov/?tab=apiaccess
        timeout: Request timeout in seconds.

    Returns:
        List of LDVariant objects, sentinel first.

    Raises:
        LDLinkError: If the API is unavailable or token is missing.
    """
    token = _resolve_ldlink_token(token)
    if token is None:
        raise LDLinkError(
            "LDlink API token required. Register free at "
            "https://ldlink.nih.gov/?tab=apiaccess, then either: "
            "(a) pass ldlink_token=..., (b) set LDLINK_TOKEN, or "
            "(c) run 'chorus setup all' to be prompted once."
        )

    import requests

    url = "https://ldlink.nih.gov/LDlinkRest/ldproxy"
    params = {
        "var": variant_id,
        "pop": population,
        "r2_d": "r2",
        "token": token,
        "genome_build": "grch38",
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
    logger.info("Found %d variants in LD (r² >= %.2f)", len(variants), r2_threshold)
    return variants


def parse_ld_response(
    text: str,
    r2_threshold: float = 0.8,
) -> list[LDVariant]:
    """Parse tab-separated LDlink LDproxy response text.

    Args:
        text: Raw TSV response from LDproxy API.
        r2_threshold: Minimum r² to include.

    Returns:
        List of LDVariant objects, sentinel first.
    """
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

        # Parse alleles: "(G/T)" or "G/T"
        alleles_str = fields[alleles_col].strip().strip("()")
        allele_parts = alleles_str.split("/")
        ref = allele_parts[0] if len(allele_parts) >= 1 else ""
        alt = allele_parts[1] if len(allele_parts) >= 2 else ""

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
