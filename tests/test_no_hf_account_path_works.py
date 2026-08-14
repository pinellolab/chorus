"""A new user without a HuggingFace account must have a path to a working install.

The TLDR's first instruction is `chorus setup`, which **halts before building anything** if no HF
token resolves (`_setup_all.py:71-83`) — correct, because AlphaGenome is gated and failing on the
last oracle after 85 GiB of downloads would be worse. But read straight through, the README implied
the account was mandatory for *chorus*, when it is mandatory only for *one of nine oracles*.

The only escape hatch the TLDR offered was `--no-weights`, which skips the gate by downloading no
weights at all — so the reader who takes it cannot run a single snippet on the page. Meanwhile
`chorus setup --oracle enformer` needs no token, builds in ~14 GiB, and runs every snippet in the
TLDR, because `main.py:93` scopes the prompt to requests that actually include `alphagenome`.

That asymmetry — a gate that is per-oracle in the code and read as global in the docs — is the kind
of thing that loses a user at step 2, so both halves are pinned here: the scoping in the code, and
the fact that the docs say so.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "chorus" / "cli" / "main.py"
SETUP_ALL = REPO / "chorus" / "cli" / "_setup_all.py"
README = REPO / "README.md"


def _guarded_token_calls(path: Path) -> list[tuple[int, str | None]]:
    """(lineno, enclosing-if-source) for every `resolve_hf_token` call in the file.

    `None` means the call is not inside an `if` at all, i.e. unconditional.
    """
    src = path.read_text()
    tree = ast.parse(src)

    # map each node to its nearest enclosing `If` test, walking the tree once
    enclosing: dict[int, ast.expr] = {}

    def walk(node: ast.AST, test: ast.expr | None) -> None:
        for child in ast.iter_child_nodes(node):
            inner = test
            if isinstance(node, ast.If) and child in node.body:
                inner = node.test
            if isinstance(child, ast.Call):
                fn = child.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "resolve_hf_token" and inner is not None:
                    enclosing[child.lineno] = inner
                elif name == "resolve_hf_token":
                    enclosing[child.lineno] = None  # type: ignore[assignment]
            walk(child, inner)

    walk(tree, None)
    return [
        (ln, ast.get_source_segment(src, t) if t is not None else None)
        for ln, t in sorted(enclosing.items())
    ]


def test_per_oracle_setup_only_prompts_when_alphagenome_is_requested():
    """`chorus setup --oracle enformer` must never ask for a token.

    This is the documented no-account path. If the guard on this call ever loses its `alphagenome`
    condition, the path silently becomes "every oracle needs a HuggingFace account".
    """
    calls = _guarded_token_calls(MAIN)
    assert calls, (
        f"no resolve_hf_token call found in {MAIN.name} — if the token gate moved, this guard needs "
        f"to follow it, or the no-account path is unverified"
    )
    for lineno, cond in calls:
        assert cond is not None, (
            f"{MAIN.name}:{lineno} resolves the HF token unconditionally, so "
            f"`chorus setup --oracle enformer` now demands an account it does not need"
        )
        assert "alphagenome" in cond.lower(), (
            f"{MAIN.name}:{lineno} guards the HF token prompt with `{cond}`, which does not mention "
            f"alphagenome. Only AlphaGenome is gated; prompting for the other oracles turns a "
            f"14 GiB no-account install into a blocked one."
        )


def test_bare_setup_still_gates_up_front():
    """The other half, and deliberately so: `chorus setup` *should* halt before downloading 85 GiB.

    Asserted so a future "fix" to the above does not quietly remove the up-front check and move the
    failure to the end of a multi-hour install.
    """
    calls = _guarded_token_calls(SETUP_ALL)
    assert calls, f"{SETUP_ALL.name} no longer resolves the HF token up front"
    conds = [c for _, c in calls]
    assert any(c is None or "no_weights" in c for c in conds), (
        f"the bare-setup gate is now conditional on something other than --no-weights ({conds}); it "
        f"must run for the default all-oracle install so auth fails before the downloads, not after"
    )


@pytest.mark.parametrize("claim,why", [
    ("--oracle enformer",
     "the no-account path has to be named where the gate is described, not 500 lines away"),
    ("no HuggingFace account",
     "a reader scanning for 'do I need an account' needs to hit the answer"),
])
def test_the_readme_offers_the_no_account_path_at_the_gate(claim: str, why: str):
    """The capability is useless if the reader has already given up."""
    text = README.read_text()
    i = text.index("chorus setup --no-weights")
    window = text[max(0, i - 1200):i + 600].lower()
    assert claim.lower() in window, (
        f"README does not mention {claim!r} near the HF-token gate. {why}."
    )


def test_no_weights_is_not_sold_as_a_getting_started_option():
    """`--no-weights` skips the gate and leaves nothing to predict with.

    It shipped as the *only* answer to "I don't want an account", which sends that reader to an
    install where every snippet on the page raises.
    """
    text = README.read_text()
    i = text.index("chorus setup --no-weights")
    sentence = text[i:i + 400]
    # (An earlier assertion here checked that the sentence contained "no" and "weights" -- which the
    # literal flag name `--no-weights` guarantees, so it could never fail. Removed rather than kept
    # as decoration.)
    assert "cannot predict" in sentence or "not for getting started" in sentence, (
        "the README describes `--no-weights` without saying it downloads no weights, so a reader "
        "takes it as the no-account path and lands on an install that cannot run the TLDR"
    )


def test_the_guard_would_catch_an_ungated_prompt():
    """Fails-without-fix, on synthetic source rather than by breaking the real file."""
    import tempfile

    bad = "def setup(args):\n    if not args.no_weights:\n        resolve_hf_token(None)\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(bad)
        tmp = Path(fh.name)
    try:
        calls = _guarded_token_calls(tmp)
        assert calls, "the AST walker found no resolve_hf_token call in the synthetic sample"
        _, cond = calls[0]
        assert cond is not None and "alphagenome" not in cond.lower(), (
            "the synthetic ungated sample should be flagged; if this passes, the real assertion "
            "above proves nothing"
        )
    finally:
        tmp.unlink(missing_ok=True)
