"""``rerender_examples.py`` must crash or refuse — never silently degrade (#133).

The script refreshes every shipped example HTML from its saved JSON with no GPU and
no model downloads. Two things were wrong with it, and the second is why the first
was left unfixed for months.

**It pointed at a directory that no longer exists.** ``examples/applications/``
became ``examples/walkthroughs/`` in 340f30e; this was the only file in the repo
still on the old name, so every invocation since 2026-04-21 died with
``FileNotFoundError`` before doing any work. Dead code, and anything added to it in
the meantime was inert.

**Fixing that path alone is worse than leaving it broken.** ``VariantReport
.from_dict`` does not carry the per-bin prediction arrays the IGV panel is drawn
from. A rehydrated report is structurally valid and renders without complaint — it
just has no signal tracks. Running the "fixed" script rewrote 15 shipped reports
from MB-scale to 0.01–0.02 MB with no exception, no warning, and a diff that reads
as a successful refresh.

So the path fix ships only together with a guard that compares the candidate output
against the artefact it is about to replace. Measured on the current tree, all 14
rehydratable reports come back at **0.3–1.0 %** of their incumbent size, so the
0.5 threshold is nowhere near either regime's boundary.

The guard renders to a temp file first. Writing and then checking would mean the
artefact is already destroyed by the time the check runs — and since the degraded
output is *valid*, there would be nothing to detect it by afterwards.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "rerender_examples.py"
WALKTHROUGHS = REPO / "examples" / "walkthroughs"


def _source() -> str:
    return SCRIPT.read_text()


def test_the_script_points_at_a_directory_that_exists():
    """The half that made it dead code."""
    assert SCRIPT.exists()
    assert 'REPO_ROOT / "examples" / "walkthroughs"' in _source()
    assert '"examples" / "applications"' not in _source()
    assert WALKTHROUGHS.is_dir()


def test_no_live_file_still_references_the_old_examples_path():
    """The rule ``audits/AUDIT_CHECKLIST.md`` item 207 already states, enforced.

    That item is a P0 checkbox reading: "live docs only reference
    ``examples/walkthroughs/`` and ``examples/notebooks/``. The old
    ``examples/applications/`` path must only appear in ``audits/`` historical
    snapshots." It was a manual grep nobody ran, and this script was in violation
    of it. Now it is a test.

    Dated files under ``audits/`` are exempt by that same rule — they record what
    was true when written. ``AUDIT_CHECKLIST.md`` is exempt because it *is* the
    rule, and this file because it names the old path while explaining it.
    """
    hits = subprocess.run(
        ["grep", "-rl", "examples/applications", "--include=*.py",
         "--include=*.md", "--include=*.yml", "--include=*.yaml", "."],
        cwd=str(REPO), capture_output=True, text=True,
    ).stdout.split()
    # rerender_examples.py names the old path in a comment explaining the rename.
    # Exempt here because test_the_script_points_at_a_directory_that_exists asserts
    # the actual path CONSTANT, which is the precise check — a substring scan cannot
    # tell an explanation from a live reference, and the explanation is worth having.
    exempt = {Path(__file__).name, "AUDIT_CHECKLIST.md", "rerender_examples.py"}
    live = [h for h in hits
            if not h.startswith("./audits/") and Path(h).name not in exempt]
    assert not live, f"stale examples/applications references in live files: {live}"


def test_every_html_write_goes_through_the_guard():
    """A second write path that skipped the guard would reopen the defect.

    The multi-oracle consolidator is the worst observed case — 9.47 MB to 0.01 MB,
    a 900x loss — so it must be guarded too, not just the per-oracle path.
    """
    src = _source()
    # to_html on a report object must never be called straight into a shipped path.
    unguarded = [
        line.strip() for line in src.splitlines()
        if ".to_html(output_path=" in line
        and "tmp_path" not in line
        and not line.strip().startswith("#")
    ]
    assert not unguarded, f"unguarded to_html write(s): {unguarded}"
    assert "_write_html_or_refuse(report," in src
    assert "_write_html_or_refuse(moracle," in src


def test_the_guard_renders_to_a_temp_file_before_comparing():
    src = _source()
    guard = src[src.index("def _write_html_or_refuse"):]
    guard = guard[:guard.index("\n# ---")]
    render = guard.index("to_html(output_path=tmp_path)")
    commit = guard.index("tmp_path.replace(out_path)")
    assert render < commit, "must render to temp BEFORE replacing the artefact"
    assert "tmp_path.unlink()" in guard, "temp file must be cleaned up"


# ---------------------------------------------------------------------------
# End to end, against the real shipped artefacts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_check_mode_refuses_everything_and_writes_nothing():
    """The behavioural gate: run it for real and confirm the tree is untouched.

    Marked integration because it shells out and renders every shipped report,
    not because it needs a GPU — it deliberately needs no oracle at all. Measured
    at ~1.3 s, so the mark is about it being a subprocess end-to-end run rather
    than about cost.
    """
    if not WALKTHROUGHS.is_dir():
        pytest.skip("no walkthroughs directory")

    before = {p: p.stat().st_mtime_ns
              for p in sorted(WALKTHROUGHS.rglob("*")) if p.is_file()}

    proc = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                          cwd=str(REPO), capture_output=True, text=True,
                          timeout=900)

    after = {p: p.stat().st_mtime_ns
             for p in sorted(WALKTHROUGHS.rglob("*")) if p.is_file()}
    changed = [str(p.relative_to(REPO)) for p in before
               if p in after and before[p] != after[p]]
    created = [str(p.relative_to(REPO)) for p in after if p not in before]

    assert not changed, f"--check modified {len(changed)} file(s): {changed[:5]}"
    assert not created, f"--check created file(s): {created[:5]}"
    assert not list(WALKTHROUGHS.rglob("*.rerender-tmp")), "temp files left behind"

    # It must refuse loudly and exit non-zero, not report success.
    assert "REFUSED" in proc.stderr + proc.stdout
    assert proc.returncode == 1, (
        f"expected exit 1 on refusal, got {proc.returncode}. A zero exit would "
        f"let CI treat a fully-refused run as a successful refresh."
    )
