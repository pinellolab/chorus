"""Interactive + env-based token resolution for `chorus setup`.

HuggingFace is required for AlphaGenome (gated model); LDlink is
optional and only used by ``fine_map_causal_variant``. The resolution
cascade is:

    CLI flag  ->  env var  ->  existing credential store  ->  stdin prompt

If the cascade fails and interactive input is not available (non-TTY,
e.g. CI), we fail without prompting. Callers decide whether the failure
is fatal (AlphaGenome setup) or just a warning (LDlink optional).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LDLINK_CONFIG_PATH = Path.home() / ".chorus" / "config.toml"


def _print_hf_setup_instructions(reason: str) -> None:
    """Print numbered steps to get a working HuggingFace token.

    Emitted on every HF-auth failure (rejected token, missing token
    non-TTY, etc.) so a user who hits this in CI / a script / a log
    sees the full path without needing to find the README.
    """
    print()
    print("─" * 72)
    print(f"  AlphaGenome requires a HuggingFace access token ({reason}).")
    print("─" * 72)
    print("  1. Create a read token:")
    print("       https://huggingface.co/settings/tokens")
    print("  2. Accept the model license (one-time):")
    print("       https://huggingface.co/google/alphagenome-all-folds")
    print("  3. Give the token to chorus — ANY of:")
    print("       a. chorus setup --oracle alphagenome --hf-token hf_xxx")
    print("       b. export HF_TOKEN=hf_xxx && chorus setup --oracle alphagenome")
    print("       c. mamba run -n chorus huggingface-cli login")
    print("─" * 72)
    print()


def _try_whoami() -> Optional[str]:
    """Return the HF username if an existing credential works, else None."""
    try:
        from huggingface_hub import whoami
    except Exception:
        return None
    try:
        info = whoami()
        return info.get("name") if isinstance(info, dict) else None
    except Exception:
        return None


def _persist_hf_token(token: str) -> None:
    """Persist an HF token via the standard huggingface_hub login flow.

    Writes to ``~/.huggingface/token`` (the path HF libraries read
    automatically), so subsequent chorus runs inherit the credential
    without re-prompting.
    """
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        logger.warning(
            "Could not persist HF token via huggingface_hub.login (%s). "
            "Falling back to HF_TOKEN env for this session only.",
            exc,
        )
        os.environ["HF_TOKEN"] = token


def resolve_hf_token(
    cli_token: Optional[str] = None,
    *,
    interactive: bool = True,
) -> bool:
    """Resolve a working HF token. Returns True when auth succeeds.

    Resolution order:
        1. ``cli_token`` (``--hf-token``)
        2. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars
        3. Existing credential (``huggingface_hub.whoami()``)
        4. Interactive stdin prompt (only if ``interactive`` and TTY)
    """
    if cli_token:
        _persist_hf_token(cli_token)
        user = _try_whoami()
        if user:
            logger.info("✓ HuggingFace auth ok (user: %s, via --hf-token)", user)
            return True
        logger.error(
            "HuggingFace rejected the token passed via --hf-token. "
            "Verify it at https://huggingface.co/settings/tokens and retry."
        )
        _print_hf_setup_instructions("token was rejected")
        return False

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        _persist_hf_token(env_token)
        user = _try_whoami()
        if user:
            logger.info("✓ HuggingFace auth ok (user: %s, via env)", user)
            return True
        logger.error(
            "HuggingFace rejected the token from HF_TOKEN / HUGGING_FACE_HUB_TOKEN. "
            "Verify it at https://huggingface.co/settings/tokens and retry."
        )
        _print_hf_setup_instructions("env-var token was rejected")
        return False

    user = _try_whoami()
    if user:
        logger.info("✓ HuggingFace auth ok (user: %s, already logged in)", user)
        return True

    if not interactive or not sys.stdin.isatty():
        logger.error("No HuggingFace token available and stdin is not a TTY.")
        _print_hf_setup_instructions("no token configured")
        return False

    _print_hf_setup_instructions("no token configured")
    try:
        from getpass import getpass
        token = getpass("  Paste HuggingFace token here (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not token:
        return False

    _persist_hf_token(token)
    user = _try_whoami()
    if user:
        logger.info("✓ HuggingFace auth ok (user: %s, saved via huggingface-cli)", user)
        return True
    logger.error(
        "HuggingFace rejected the provided token. "
        "Verify it at https://huggingface.co/settings/tokens and retry."
    )
    _print_hf_setup_instructions("token was rejected")
    return False


def prompt_ldlink_token(*, interactive: bool = True) -> Optional[str]:
    """Optionally prompt for an LDlink token. Non-blocking — returns None
    if the user skips.

    If provided, persist to ``~/.chorus/config.toml`` so
    ``chorus.utils.ld`` can pick it up as a fallback.
    """
    env_token = os.environ.get("LDLINK_TOKEN")
    if env_token:
        logger.info("✓ LDlink token detected in env (LDLINK_TOKEN)")
        return env_token

    existing = _read_ldlink_token_from_config()
    if existing:
        logger.info("✓ LDlink token already saved in %s", _LDLINK_CONFIG_PATH)
        return existing

    if not interactive or not sys.stdin.isatty():
        return None

    print()
    print("Optional: LDlink API token (only needed for fine_map_causal_variant).")
    print("  Register free at https://ldlink.nih.gov/?tab=apiaccess")
    print("  Press Enter to skip.")
    try:
        from getpass import getpass
        token = getpass("LDlink token (optional, hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not token:
        logger.info("LDlink token skipped (set LDLINK_TOKEN later if needed)")
        return None

    _write_ldlink_token_to_config(token)
    logger.info("✓ LDlink token saved to %s", _LDLINK_CONFIG_PATH)
    return token


def _read_ldlink_token_from_config() -> Optional[str]:
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


def _write_ldlink_token_to_config(token: str) -> None:
    _LDLINK_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if _LDLINK_CONFIG_PATH.exists():
        try:
            existing = _LDLINK_CONFIG_PATH.read_text()
        except Exception:
            existing = ""

    if "[tokens]" in existing:
        # Replace existing ldlink line or append under [tokens].
        import re
        new = re.sub(
            r'(?m)^ldlink\s*=.*$', f'ldlink = "{token}"', existing
        )
        if new == existing:
            new = existing.replace("[tokens]", f'[tokens]\nldlink = "{token}"', 1)
        _LDLINK_CONFIG_PATH.write_text(new)
    else:
        sep = "" if existing.endswith("\n") or not existing else "\n"
        _LDLINK_CONFIG_PATH.write_text(
            existing + sep + f'[tokens]\nldlink = "{token}"\n'
        )
