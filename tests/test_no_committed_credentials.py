"""No tracked file may contain a credential — including the unprefixed shapes.

A live LDlink API token sat in `audits/2026-04-23_v23_scorched_earth/report.md` for nearly four
months, inside a paragraph headed "Reminder / hygiene" that asserted "No copy was written to any
on-disk location by me during the audit". The HF token on the very next line *was* redacted; the
LDlink one was not.

**Why four secret sweeps missed it.** Every sweep, including the v0.7.3 release audit's
"0 `hf_…` tokens, 0 AWS keys anywhere", searched for *prefixed* patterns — `hf_`, `AKIA`,
`ghp_`. An LDlink token is **twelve bare hex characters with no prefix**, so none of those
greps could ever have matched it, and each clean result was reported as "no secrets" rather than
"no secrets of the shapes we grep for". That gap is the actual defect this file closes; the leak
was only its first symptom.

An unprefixed 12-hex string cannot be matched on shape alone — it is also a short git sha, a CRC,
a colour table, an ENCODE id fragment. So the rule here is **contextual**: a hex run is a finding
when a nearby word says it is a credential. That is what the leak looked like, and it keeps the
guard from drowning in the thousands of legitimate hashes this repo commits.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Prefixed credentials, matchable on shape alone.
PREFIXED = {
    "huggingface": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github-pat": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    "openai": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "slack": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
}

#: An unprefixed secret is only identifiable from context: a credential word within ~40 characters
#: before a bare hex/base62 run. This is the exact shape the LDlink leak had.
CREDENTIAL_WORD = r"(?:token|api[_ -]?key|apikey|secret|password|passwd|credential|bearer)"
CONTEXTUAL = re.compile(
    CREDENTIAL_WORD + r"[^\n]{0,40}?[`'\"\(]([0-9a-f]{12,}|[A-Za-z0-9]{20,})[`'\"\)]",
    re.I,
)

#: Redaction markers — a value that has been defused, not a live secret.
REDACTED = re.compile(
    r"redact|REDACTED|\betc\b|\.\.\.|…|xxx+|<[^>]+>|\bexample\b|\byour[_ -]|\bhf_xxx|placeholder"
    r"|\bhf_\.\.\.|NOT_A_REAL|dummy|fake|sha256|md5\b",
    re.I,
)

#: Files whose whole purpose is to describe credential shapes.
ALLOW_PATHS = {
    "tests/test_no_committed_credentials.py",
}

#: Exempt from the *contextual* heuristic only — these files deliberately hold synthetic
#: credential-shaped fixtures (e.g. ``SECRET = "SUPERSECRETLDTOKEN123"``) to prove the redaction
#: paths work. They stay in scope for every prefixed pattern above, because a real leaked token
#: has a prefix and a test file is not a licence to commit one.
CONTEXTUAL_EXEMPT = {
    "tests/test_credentials_never_reach_an_error_message.py",
}


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    keep = {".md", ".py", ".ipynb", ".yml", ".yaml", ".json", ".toml", ".txt", ".sh", ".cfg", ".js"}
    return [REPO / p for p in out.split("\0")
            if p and Path(p).suffix in keep and p not in ALLOW_PATHS]


FILES = _tracked_text_files()


@pytest.mark.parametrize("kind,pattern", sorted(PREFIXED.items()))
def test_no_prefixed_credential_is_committed(kind: str, pattern: re.Pattern):
    hits = []
    for path in FILES:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in pattern.finditer(line):
                if REDACTED.search(line):
                    continue
                hits.append(f"{path.relative_to(REPO)}:{i}  …{m.group(0)[:6]}… ({kind})")
    assert not hits, f"{kind} credential(s) in tracked files:\n  " + "\n  ".join(hits)


def test_no_credential_shaped_value_sits_next_to_a_credential_word():
    """The unprefixed case — the shape the LDlink leak actually had."""
    hits = []
    for path in FILES:
        if str(path.relative_to(REPO)) in CONTEXTUAL_EXEMPT:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = CONTEXTUAL.search(line)
            if not m or REDACTED.search(line):
                continue
            hits.append(f"{path.relative_to(REPO)}:{i}  (…{len(m.group(1))} chars after "
                        f"a credential word)")
    assert not hits, (
        "a credential-shaped value sits next to a word calling it a credential:\n  "
        + "\n  ".join(hits)
        + "\nIf it is genuinely not a secret, add a redaction marker or rephrase the line. "
          "If it is, rotate it — removing it from the tree does not undo git history."
    )


@pytest.mark.parametrize("leak", [
    "The LDlink token the user pasted this session (`5b19f9d3d067`) and",
    'api_key="a1b2c3d4e5f6"',
    "password: `hunter2hunter2hunter2`",
])
def test_the_guard_catches_the_unprefixed_shapes(leak: str):
    """Fails-without-fix. The first string is the leak that actually shipped, verbatim."""
    assert CONTEXTUAL.search(leak) and not REDACTED.search(leak), \
        f"guard no longer catches: {leak}"


@pytest.mark.parametrize("benign", [
    "the tag was force-moved to `a79feb0` after publication",
    "verified via sha256 `9fe92856bd189042207a8696f96758c53ea5cdd6`",
    "set HF_TOKEN to your token, e.g. `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`",
    "The LDlink token (`redacted 2026-08-13 — treat as compromised`) and",
    "token (`hf_yzF…` — redacted in logs)",
])
def test_the_guard_leaves_hashes_and_redactions_alone(benign: str):
    """The other half: this repo commits thousands of legitimate hashes."""
    flagged = CONTEXTUAL.search(benign) and not REDACTED.search(benign)
    assert not flagged, f"false positive on a benign line: {benign}"
