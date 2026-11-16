"""A transport error must not print the caller's credential.

``requests`` builds its exception messages from the full request URL, and the
LDlink token travels as a ``token=`` query parameter. So every failure mode —
timeout, 429, 401, a 503 raised by ``raise_for_status`` — put the user's real
token verbatim into ``LDLinkError``. From there it reaches:

  * ``chorus/mcp/server.py`` as ``{"error": str(exc)}``, i.e. into the agent
    transcript;
  * ``f"Could not resolve {rsid!r} via LDlink: {exc}"``;
  * any notebook output or log that captures the exception.

That is a code path, not an accident of one session, so it would have leaked for
every user on every LDlink outage. Two HuggingFace tokens were being rotated at
the time this was found for the much less systematic reason of having been pasted
once.

The redactor is deliberately belt-and-braces: it removes the secret it was handed
AND blanks any ``token=`` parameter, so a credential that arrived by a route the
function was not told about is still not printed. These tests check both halves,
because the second is the one that survives refactoring.

A clean message was not enough (2026-11-15 pre-release audit, P1-6). The redacted
``LDLinkError`` was raised ``from exc``, so the unredacted requests exception rode
along as its cause, and ``_safe_tool``'s ``logger.exception`` printed the whole chain
to the MCP server's stderr -- which clients keep as a log file -- and
``CHORUS_MCP_DEBUG=1`` returned it in ``payload["traceback"]``. urllib3 also logs the
request line, query string included, at DEBUG. The tests at the bottom drive an
LDlink outage through the MCP layer and check all three places.
"""
from __future__ import annotations

import json
import logging
import traceback

import pytest

from chorus.utils.ld import _redact

SECRET = "SUPERSECRETLDTOKEN123"
FAKE_TOKEN = "abc123def456"  # fake, shaped like an LDlink token; not a credential


@pytest.mark.parametrize("message", [
    # The exact shapes requests produces.
    "HTTPSConnectionPool(host='ldlink.nih.gov', port=443): Max retries exceeded "
    f"with url: /LDlinkRest/ldproxy?var=rs12740374&pop=CEU&token={SECRET}&genome_build=grch38",
    f"503 Server Error: Service Unavailable for url: http://x/LDlinkRest/ldproxy?a=1&token={SECRET}",
    f"401 Client Error: Unauthorized for url: https://ldlink.nih.gov/x?token={SECRET}",
    # And a body that echoes the request back.
    f"error: invalid token {SECRET}",
])
def test_the_secret_is_removed_from_every_message_shape(message: str):
    out = _redact(message, SECRET)
    assert SECRET not in out, out
    assert "<redacted>" in out


def test_a_token_the_redactor_was_not_told_about_is_still_blanked():
    """The half that survives refactoring.

    If a future change stops threading the token into the call, the query-parameter
    rule still catches it. Without this, "we pass the secret in" becomes a silent
    precondition.
    """
    out = _redact("Max retries exceeded with url: /x?var=rs1&token=UNKNOWN_TOKEN_XYZ&pop=CEU")
    assert "UNKNOWN_TOKEN_XYZ" not in out, out
    assert "token=<redacted>" in out


def test_redaction_leaves_everything_diagnostic_intact():
    """A redactor that eats the useful part of the message gets removed by the next
    person to debug an LDlink outage."""
    msg = ("HTTPSConnectionPool(host='ldlink.nih.gov', port=443): Max retries exceeded "
           f"with url: /LDlinkRest/ldproxy?var=rs12740374&pop=CEU&token={SECRET}")
    out = _redact(msg, SECRET)
    for keep in ("ldlink.nih.gov", "Max retries exceeded", "rs12740374", "pop=CEU"):
        assert keep in out, f"redaction destroyed {keep!r}, which is what you debug with"


def test_no_secret_survives_a_real_failure_end_to_end():
    """The integration form: drive the real function into a transport error."""
    from chorus.utils.ld import LDLinkError, fetch_ld_variants

    try:
        fetch_ld_variants("rs12740374", token=SECRET, timeout=0.001)
    except LDLinkError as exc:
        assert SECRET not in str(exc), str(exc)
    except Exception as exc:  # any other failure must also be clean
        assert SECRET not in str(exc), f"{type(exc).__name__}: {exc}"
    else:
        pytest.skip("LDlink answered within 1 ms, so no error path was exercised")


def test_the_error_paths_redact_rather_than_interpolating_raw():
    """Guard the pattern, not just today's two call sites.

    A third ``raise LDLinkError(f"... {exc}")`` added later would reintroduce the
    leak while every test above still passed.
    """
    import inspect
    import re

    from chorus.utils import ld

    src = inspect.getsource(ld)
    # Any interpolation of an exception or a response body into an error message
    # must go through _redact.
    for m in re.finditer(r"LDLinkError\(\s*f?\"[^\"]*\{(exc|text[^}]*)\}", src):
        snippet = src[max(0, m.start() - 120):m.end() + 120]
        assert "_redact" in snippet, (
            f"an LDLinkError interpolates {m.group(1)!r} without _redact:\n{snippet}"
        )


# --- The chain, the server log and the debug payload (P1-6) ------------------------


def _ldlink_outage(monkeypatch):
    """Make ``requests.get`` fail as an LDlink outage does: urllib3 logs the request
    line, then a 503 whose message is the full request URL, token included."""
    import requests

    def fake_get(url, params=None, timeout=None):
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        # urllib3.connectionpool's own DEBUG line, emitted for every request.
        logging.getLogger("urllib3.connectionpool").debug(
            '%s://%s:%s "%s %s %s" %s %s', "https", "ldlink.nih.gov", 443, "GET",
            f"/LDlinkRest/ldproxy?{query}", "HTTP/1.1", 503, None)
        raise requests.exceptions.HTTPError(
            f"503 Server Error: Service Unavailable for url: {url}?{query}")

    monkeypatch.setattr(requests, "get", fake_get)


@pytest.mark.parametrize("entry", ["fetch_ld_variants", "_resolve_rsid_to_position"])
def test_the_traceback_of_an_ldlink_outage_does_not_carry_the_token(monkeypatch, entry):
    """The chain itself must be clean, whoever prints it -- a notebook, pytest, a logger."""
    from chorus.mcp import server
    from chorus.utils import ld

    _ldlink_outage(monkeypatch)
    try:
        if entry == "fetch_ld_variants":
            ld.fetch_ld_variants("rs12740374", token=FAKE_TOKEN)
        else:
            server._resolve_rsid_to_position("rs12740374", ldlink_token=FAKE_TOKEN)
    except (ld.LDLinkError, ValueError):
        tb = traceback.format_exc()
    else:
        pytest.fail("an LDlink outage must raise")
    assert FAKE_TOKEN not in tb, tb
    assert "503 Server Error" in tb, "the redacted message is what you debug an outage with"


@pytest.mark.parametrize("debug_logger", [None, "urllib3", "urllib3.connectionpool"],
                         ids=["root", "urllib3", "urllib3.connectionpool"])
def test_an_ldlink_outage_in_an_mcp_tool_leaks_into_no_log_and_no_payload(
        monkeypatch, caplog, debug_logger):
    """The audit's reproduction: an rsID tool call while LDlink is down, with the
    server logging at DEBUG (root, or the urllib3 loggers directly) and
    CHORUS_MCP_DEBUG=1."""
    from chorus.mcp import server

    _ldlink_outage(monkeypatch)
    monkeypatch.setenv("CHORUS_MCP_DEBUG", "1")
    caplog.set_level(logging.DEBUG)
    if debug_logger:
        caplog.set_level(logging.DEBUG, logger=debug_logger)
    levels = {n: logging.getLogger(n).level for n in ("urllib3", "urllib3.connectionpool")}

    @server._safe_tool
    def rsid_tool(position, ldlink_token=None):  # the shape of the three rsID tools
        return server._resolve_rsid_to_position(position, ldlink_token=ldlink_token)

    payload = rsid_tool("rs12740374", ldlink_token=FAKE_TOKEN)

    assert payload["error_type"] == "ValueError", payload
    assert "MCP tool rsid_tool failed" in caplog.text, "nothing was logged, so nothing was checked"
    assert FAKE_TOKEN not in caplog.text, caplog.text
    assert FAKE_TOKEN not in payload["traceback"], payload["traceback"]
    assert FAKE_TOKEN not in json.dumps(payload)
    assert "token=<redacted>" in payload["traceback"]
    # The hold is released afterwards, or urllib3 debugging stays off for the process.
    assert {n: logging.getLogger(n).level for n in levels} == levels


def test_safe_tool_scrubs_credentials_from_what_it_logs_and_returns(monkeypatch, caplog):
    """The second barrier, for a credential that reaches a traceback by a route nobody
    has found yet: the env-var values, a ``*_token`` argument, any ``token=`` parameter."""
    from chorus.mcp import server

    hf = "hf_" + "A1b2C3d4E5" * 3 + "f6G7"  # fake, assembled so no scanner sees a token
    passed = "passedtoken" + "456789"
    monkeypatch.setenv("HF_TOKEN", hf)
    monkeypatch.setenv("LDLINK_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CHORUS_MCP_DEBUG", "1")

    @server._safe_tool
    def leaky(ldlink_token=None):
        try:
            raise ConnectionError("GET https://example.org/x?var=1&token=querytoken987 failed")
        except ConnectionError as cause:
            raise RuntimeError(f"auth failed: {hf} {FAKE_TOKEN} {ldlink_token}") from cause

    with caplog.at_level(logging.ERROR, logger="chorus.mcp.server"):
        payload = leaky(ldlink_token=passed)

    assert "MCP tool leaky failed" in caplog.text, "nothing was logged, so nothing was checked"
    for secret in (hf, FAKE_TOKEN, passed, "querytoken987"):
        assert secret not in caplog.text, secret
        assert secret not in json.dumps(payload), secret
    # Redaction must not eat the diagnosis.
    assert "auth failed" in payload["error"]
    assert "ConnectionError" in payload["traceback"] and "example.org" in caplog.text
