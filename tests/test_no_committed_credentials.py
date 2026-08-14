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

#: Files skipped entirely. Empty on purpose: a security guard that exempts a file from its own
#: scan cannot report a credential pasted into that file, and this one exempted *itself* while
#: carrying the real leaked token as a fixture. Use CONTEXTUAL_EXEMPT below instead, which keeps
#: every prefixed pattern in force.
ALLOW_PATHS: set[str] = set()

#: Exempt from the *contextual* heuristic only — these files deliberately hold synthetic
#: credential-shaped fixtures (e.g. ``SECRET = "SUPERSECRETLDTOKEN123"``) to prove the redaction
#: paths work. They stay in scope for every prefixed pattern above, because a real leaked token
#: has a prefix and a test file is not a licence to commit one.
CONTEXTUAL_EXEMPT = {
    "tests/test_credentials_never_reach_an_error_message.py",
    # This file's own fixtures are credential-shaped by design. It stays in scope for every
    # PREFIXED pattern, so a real `hf_…` pasted here is still caught.
    "tests/test_no_committed_credentials.py",
}


#: Suffixes that are genuinely binary. Everything else is opened and scanned.
#:
#: This used to be the other way round — an allowlist of "text" suffixes — and that inverted the
#: default in the wrong direction for a security guard. Measured on this repo it opened 655 of 869
#: tracked files and **never opened 214**, including **59 `.log` files and 20 `.html` reports**:
#: precisely the artefacts a token leaks into, since a log captures whatever was in the environment
#: and a rendered report captures whatever was in a URL. A manual sweep of those 214 found nothing,
#: so no leak was missed — but a guard that cannot see the most likely hiding places is a claim of
#: coverage rather than coverage, which is the exact failure this file exists to correct.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".npz", ".npy", ".gz", ".zip", ".tar", ".bz2", ".xz",
    ".h5", ".hdf5", ".pt", ".pth", ".safetensors", ".bin", ".pkl", ".2bit",
    ".woff", ".woff2", ".ttf", ".eot",
}


def _tracked_text_files() -> list[Path]:
    """Every tracked file that is not obviously binary, so the default is *scan*."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [REPO / p for p in out.split("\0")
            if p and Path(p).suffix.lower() not in BINARY_SUFFIXES and p not in ALLOW_PATHS]


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


def test_the_guard_opens_the_file_types_a_token_actually_leaks_into():
    """Coverage is part of the guarantee, so it is asserted rather than assumed.

    The first version filtered by an allowlist of "text" suffixes and so never opened 59 `.log`
    files or 20 `.html` reports — the two places a credential is most likely to end up, because a log
    captures the environment and a rendered report captures URLs. Scanning is the default now; only
    known-binary suffixes are skipped.
    """
    suffixes = {p.suffix.lower() for p in FILES}
    # `present` comes from raw `git ls-files`, NOT from _tracked_text_files(): both sides of this
    # check used to derive from the same filter, so putting `.log` back into BINARY_SUFFIXES -- the
    # exact regression this test exists to prevent -- emptied `present` and skipped the assertion
    # instead of failing it.
    all_tracked = [REPO / f for f in subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\0") if f]
    checked = []
    for risky in (".log", ".html", ".tsv", ".ini"):
        present = [p for p in all_tracked if p.suffix.lower() == risky]
        if not present:
            continue  # e.g. .log: #198 removed every tracked one, so there is nothing to cover
        checked.append(risky)
        assert risky in suffixes, (
            f"{len(present)} tracked {risky} files exist and the guard does not open them; a token "
            f"in one would go unreported"
        )

    assert checked, (
        "none of the risky suffixes exist in the tree any more, so the loop above asserted nothing. "
        "Add a suffix that does exist, or this half of the coverage guarantee is decoration."
    )

    tracked = len([f for f in subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                                             capture_output=True, text=True,
                                             check=True).stdout.split("\0") if f])
    assert len(FILES) / tracked > 0.80, (
        f"the guard opens only {len(FILES)} of {tracked} tracked files "
        f"({100 * len(FILES) / tracked:.0f}%). It was 75% when a live token sat undetected in a "
        f"file it did read — shrinking coverage is the wrong direction."
    )


@pytest.mark.parametrize("leak", [
    # Same SHAPE as the value that leaked (12 bare hex, backticked, after the word "token"),
    # deliberately NOT the value itself. The first version of this test pasted the real token
    # back into the repo -- in the one file whose job is to stop exactly that -- and then relied
    # on the self-exemption below to keep passing. A fixture does not need the true secret to
    # prove the pattern matches.
    "The LDlink token the user pasted this session (`0123456789ab`) and",
    'api_key="a1b2c3d4e5f6"',
    "password: `hunter2hunter2hunter2`",
])
def test_the_guard_catches_the_unprefixed_shapes(leak: str):
    """Fails-without-fix, on the *shape* that shipped rather than the value.

    The first string reproduces the construction that leaked -- twelve bare hex characters next to
    the word "token" -- using a synthetic value. It said "verbatim" until #197 replaced the real
    token with `0123456789ab`, leaving a docstring that invited a maintainer to paste the true
    secret back into the one file whose job is to stop exactly that.
    """
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
