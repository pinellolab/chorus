"""`chorus setup all` — orchestrates end-to-end setup of every oracle.

Flow:
    1. Resolve the HuggingFace token (blocking — no fallback). If the
       user cannot produce a working token, halt BEFORE any env build
       or download runs, because AlphaGenome cannot proceed without it
       and the user asked for "all".
    2. Optionally prompt for an LDlink token (non-blocking).
    3. For each oracle, build the conda env, pre-download weights,
       background CDFs, and (once) the hg38 reference, then write the
       setup-complete marker.

The single-oracle flow (``chorus setup <oracle>``) lives in
``main.setup_environments``; ``setup all`` is intentionally a separate
entry so its stricter gating is obvious from the call graph.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Oracles excluded from the default ``chorus setup`` flow because they're
# alternative backends for an oracle that's already installed by default.
# Currently empty — both AlphaGenome backends (JAX `alphagenome` and
# PyTorch `alphagenome_pt`) install by default so Mac users get MPS
# access without an extra setup step. The infrastructure stays in place
# for future opt-in oracles. The ``--include-alternative-backends`` flag
# is preserved as a no-op for now to keep CLI shape stable.
_SKIP_FROM_DEFAULT_SETUP: set[str] = set()


def setup_all_oracles(args) -> int:
    """Implementation of ``chorus setup --oracle all``."""
    from ..core.environment import EnvironmentManager, EnvironmentRunner
    from ..core.weights_probe import write_setup_marker
    from ._setup_prefetch import prefetch_for_oracle
    from ._tokens import prompt_ldlink_token, resolve_hf_token

    manager = EnvironmentManager()
    all_oracles = manager.list_available_oracles()
    if not all_oracles:
        logger.error("No oracle environment definitions found.")
        return 1

    # Filter alternative-backend oracles unless --include-alternative-backends
    # was passed. The legacy --include-experimental alias keeps the same
    # behavior for anyone who scripted against the older name.
    include_alt = getattr(args, "include_alternative_backends", False) or getattr(
        args, "include_experimental", False
    )
    if include_alt:
        oracles = all_oracles
    else:
        oracles = [o for o in all_oracles if o not in _SKIP_FROM_DEFAULT_SETUP]
        skipped = sorted(set(all_oracles) - set(oracles))
        if skipped:
            logger.info(
                "Skipping alternative-backend oracle(s): %s "
                "(install with `chorus setup --oracle <name>` or pass "
                "--include-alternative-backends to enable in the default "
                "flow). These are opt-in only because of disk size, not "
                "stability — see the README's 'Two AlphaGenome backends' "
                "section for details.",
                ", ".join(skipped),
            )

    # HF token gate — must resolve BEFORE we start downloading 10+ GB of
    # env + weights only to fail on the last oracle.
    if not args.no_weights:
        if not resolve_hf_token(
            cli_token=getattr(args, "hf_token", None),
            interactive=True,
        ):
            logger.error(
                "`chorus setup all` halted: a working HuggingFace token is "
                "required for AlphaGenome. Nothing was downloaded. "
                "Set HF_TOKEN, run 'huggingface-cli login', or pass "
                "--hf-token and retry."
            )
            return 1

    # Non-blocking LDlink prompt.
    prompt_ldlink_token(interactive=True)

    runner = EnvironmentRunner(manager)
    success_count = 0
    for oracle in oracles:
        logger.info(f"\n=== Setting up {oracle} ===")
        if args.force:
            from ..core.weights_probe import setup_marker_path
            stale = setup_marker_path(oracle)
            if stale.exists():
                stale.unlink()
        if not manager.create_environment(oracle, force=args.force):
            logger.error(f"✗ Failed to build env for {oracle}")
            continue
        logger.info(f"✓ env for {oracle}")

        if args.no_weights and args.no_backgrounds and args.no_genome:
            logger.info(f"Skipping all data prefetch for {oracle}")
            success_count += 1
            continue

        ok, errors = prefetch_for_oracle(
            oracle,
            runner,
            skip_weights=args.no_weights,
            skip_backgrounds=args.no_backgrounds,
            skip_genome=args.no_genome,
            full_chrombpnet=getattr(args, "all_chrombpnet", False),
        )
        if not ok:
            logger.error(f"✗ prefetch failed for {oracle}:")
            for err in errors:
                logger.error(f"  - {err}")
            continue

        if args.no_weights:
            logger.info(
                f"✓ {oracle} env ready (weights skipped — setup marker NOT written)"
            )
        else:
            write_setup_marker(oracle)
            logger.info(f"✓ {oracle} ready")
        success_count += 1

    logger.info(f"\nSetup all complete: {success_count}/{len(oracles)} oracles ready.")
    return 0 if success_count == len(oracles) else 1
