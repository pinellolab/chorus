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
"""
from __future__ import annotations

import pytest

from chorus.utils.ld import _redact

SECRET = "SUPERSECRETLDTOKEN123"


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
