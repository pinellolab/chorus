"""`chorus-mcp` must reject unknown flags and describe itself from the code.

Before this, `main()` was `if "--help" in sys.argv or "-h" in sys.argv:` followed by `mcp.run()`.
Two consequences:

* **`chorus-mcp --port 9000` started a stdio server.** Any flag, or any typo, was swallowed — the
  user got a silently-not-what-they-asked-for server rather than an error. There was no parser at
  all, so `host`/`port`/transport were undocumented *and* unsettable through the CLI.
* **The tool list was hand-maintained and had drifted twice.** A v27 audit found it missing
  `discover_variant` and `fine_map_causal_variant`; by v0.7.3 it still announced "Tools provided
  (22)" while 24 were registered (`recommend_alphagenome_backend` and `score_ism` missing). The
  count is now derived from the server object, so it cannot disagree.

The oracle count in `_INSTRUCTIONS` disagreed with `list_oracles`' docstring which disagreed with
`ORACLE_SPECS` — 7 vs 8 vs 9 — and that string is the system prompt an assistant reads, so it is
worth pinning too.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "chorus" / "mcp" / "server.py"


def test_the_advertised_tool_count_equals_what_is_registered():
    from chorus.mcp.server import registered_tool_names

    names = registered_tool_names()
    decorated = len(re.findall(r"^@mcp\.tool\(\)", SERVER.read_text(), re.M))
    assert names, "no tools discovered — the help text would silently list none"
    assert len(names) == decorated, (
        f"{len(names)} tools registered but {decorated} `@mcp.tool()` decorators in the source"
    )


def test_help_is_generated_not_hardcoded():
    """A literal count in the help text is the thing that drifted; there must not be one."""
    text = SERVER.read_text()
    start = text.index("def main(")
    body = text[start:]
    assert not re.search(r"Tools provided \(\d+\)", body), (
        "the help text hardcodes a tool count again — derive it from the server instead"
    )


@pytest.mark.parametrize("flag", ["--port", "--transport", "--bogus"])
def test_an_unknown_flag_is_rejected_rather_than_swallowed(flag):
    """Fails-without-fix: the old `sys.argv` sniff started a stdio server for all of these."""
    proc = subprocess.run(
        ["chorus-mcp", flag, "9000"], capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode != 0, (
        f"`chorus-mcp {flag} 9000` exited 0 — an unrecognised flag must not start a server"
    )
    assert "unrecognized arguments" in proc.stderr or "usage:" in proc.stderr, proc.stderr


def test_help_exits_zero_and_names_the_undocumented_env_vars():
    proc = subprocess.run(["chorus-mcp", "--help"], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for var in ("CHORUS_NO_TIMEOUT", "CHORUS_MCP_OUTPUT_DIR", "CHORUS_MCP_DEBUG",
                "FASTMCP_TRANSPORT"):
        assert var in out, f"{var} is read by the code but absent from --help"
    assert "model-load" in out, (
        "CHORUS_NO_TIMEOUT also disables model-load timeouts (chorus/core/base.py); saying only "
        "'prediction timeouts' understates it"
    )
    assert "client" in out.lower(), (
        "CHORUS_MCP_OUTPUT_DIR defaults relative to the client's working directory, captured at "
        "startup — that surprise belongs in the help"
    )


def test_the_system_prompt_oracle_count_matches_the_registry():
    from chorus.mcp.server import ORACLE_SPECS, _INSTRUCTIONS

    assert len(ORACLE_SPECS) == 9, f"ORACLE_SPECS has {len(ORACLE_SPECS)} entries"
    assert "9 registered" in _INSTRUCTIONS, (
        "_INSTRUCTIONS is the system prompt an assistant reads; it must not undercount the oracles"
    )
    for named in ("Cherimoya", "EPInformer-seq", "AlphaGenome"):
        assert named in _INSTRUCTIONS, f"{named} missing from the system prompt"
