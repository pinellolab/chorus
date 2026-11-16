"""The quickstart's conservation cell must not start a ~70 GB download on "Run All".

``single_oracle_quickstart.ipynb`` is the notebook ``examples/notebooks/README.md`` sends
first-time users to ("15 min", "runs fully on CPU with 8 GB RAM"). Its *Conservation Tracks*
cell, added in 3d0b6b1, called ``build_variant_report(..., show_conservation=True)``
unconditionally, and on a fresh data dir that fetches seven genome-wide hg38 bigwigs: GPN-Star
entropy, its four per-allele LLR files, phyloP and phastCons, 66.0 GiB (~71 GB) on disk. In the
2026-11-15 pre-release audit (P1-2) the prescribed ``nbconvert`` run was killed at 541 s, still
inside that cell.

The cell now sits behind ``RUN_CONSERVATION = False`` and, with the gate shut, prints the
``chorus conservation download`` commands instead. The regression to expect is someone flipping
the flag to commit outputs that show the report: on a warm cache that executes cleanly and
ships the download again, so no re-execution would ever notice.

Hermetic: the scan reads the committed JSON, and only the gate-shut branch is run, against
``list_tracks()`` pointed at a ``tmp_path``. No oracle, no network, no real data dir.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
from pathlib import Path

import pytest

from chorus.analysis import conservation
from chorus.cli import _conservation

NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "examples" / "notebooks" / "single_oracle_quickstart.ipynb")
FLAG = "RUN_CONSERVATION"


def _parse(source: str) -> ast.Module:
    """Parse a cell, blanking IPython line magics (``%matplotlib inline``) first."""
    return ast.parse("\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    ))


def _asks_for_conservation(kw: ast.keyword) -> bool:
    return kw.arg == "show_conservation" and not (
        isinstance(kw.value, ast.Constant) and kw.value.value is False)


def _conservation_cells() -> list[tuple[int, ast.Module]]:
    """(index, tree) of every code cell that passes ``show_conservation``."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = _parse("".join(cell["source"]))
        if any(isinstance(n, ast.keyword) and _asks_for_conservation(n) for n in ast.walk(tree)):
            found.append((i, tree))
    assert found, (
        f"no code cell in {NOTEBOOK.name} passes show_conservation any more. If the section was "
        "removed on purpose, delete this file; if it was rewritten, this scan no longer finds it."
    )
    return found


def _flag_values(tree: ast.Module) -> list[ast.expr]:
    return [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == FLAG for t in node.targets)]


def test_every_show_conservation_call_waits_for_the_flag():
    for i, tree in _conservation_cells():
        gated = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == FLAG:
                gated.update(id(n) for stmt in node.body for n in ast.walk(stmt))
        ungated = [kw.value.lineno for kw in ast.walk(tree)
                   if isinstance(kw, ast.keyword) and _asks_for_conservation(kw)
                   and id(kw) not in gated]
        assert not ungated, (
            f"{NOTEBOOK.name} cell {i} passes show_conservation outside `if {FLAG}:` (cell "
            f"line(s) {ungated}); on a fresh data dir that is a ~70 GB download the first time "
            "'Run All' reaches it."
        )


def test_the_flag_ships_off():
    for i, tree in _conservation_cells():
        values = _flag_values(tree)
        assert len(values) == 1, (
            f"{NOTEBOOK.name} cell {i}: expected one top-level `{FLAG} = ...`, found {len(values)}")
        assert isinstance(values[0], ast.Constant) and values[0].value is False, (
            f"{NOTEBOOK.name} cell {i} ships `{FLAG} = {ast.unparse(values[0])}`. Set it back to "
            "False before committing: a warm cache hides the ~70 GB download a fresh install pays."
        )


@pytest.mark.parametrize("cached", [False, True], ids=["fresh-data-dir", "everything-cached"])
def test_the_shut_gate_prints_commands_the_cli_accepts(monkeypatch, capsys, tmp_path, cached):
    cells = _conservation_cells()
    assert len(cells) == 1, f"expected one conservation cell, found cells {[i for i, _ in cells]}"
    (i, tree), = cells
    values = _flag_values(tree)
    if not (len(values) == 1 and isinstance(values[0], ast.Constant) and values[0].value is False):
        pytest.fail(f"cell {i} does not ship with {FLAG} = False; running it here would build "
                    "the report for real (see test_the_flag_ships_off)")

    real_list_tracks = conservation.list_tracks
    tracks = real_list_tracks(tmp_path)
    if cached:
        for info in tracks.values():
            info["path"].parent.mkdir(parents=True, exist_ok=True)
            info["path"].touch()
    monkeypatch.setattr(conservation, "list_tracks", lambda: real_list_tracks(tmp_path))

    namespace: dict = {}
    exec(compile(tree, f"{NOTEBOOK.name}[{i}]", "exec"), namespace)
    out = capsys.readouterr().out
    assert "nothing was downloaded" in out and "conservation_report" not in namespace

    commands = [shlex.split(line.split("#", 1)[0]) for line in out.splitlines()
                if line.strip().startswith("chorus conservation download")]
    if cached:
        assert not commands, f"every track is cached, yet the cell still prints:\n{out}"
        assert f"All {len(tracks)} conservation bigwigs are cached" in out
        return

    # Parse what the cell prints with the real subcommand, so a renamed flag or track fails here
    # rather than in a user's terminal.
    parser = argparse.ArgumentParser(prog="chorus")
    _conservation.register_conservation_subcommand(parser.add_subparsers(dest="command"))
    named, download_all = set(), False
    for argv in commands:
        try:
            args = parser.parse_args(argv[1:])
        except SystemExit:  # argparse exits on an unknown option
            pytest.fail(f"the cell prints `{shlex.join(argv)}`, which `chorus` rejects")
        assert args.func is _conservation.conservation_download, argv
        download_all |= args.all
        if args.track:
            named.update(_conservation._TRACK_ALIASES.get(args.track, [args.track]))
    assert named == set(tracks), (
        f"the cell names {sorted(named)}, but show_conservation=True reads {sorted(tracks)}")
    assert download_all, "the cell no longer prints the one-line `--all` form"

    # The ~70 GB the cell quotes is a literal; the sizes it lists come from the track table.
    quoted = re.search(r"~(\d+) GB for all", out)
    assert quoted, out
    total = sum(float(re.fullmatch(r"~([\d.]+) GB", info["size_note"]).group(1))
                for info in tracks.values())
    assert abs(total - float(quoted.group(1))) <= 3.5, (
        f"the cell says ~{quoted.group(1)} GB, but the track table's size notes add up to "
        f"{total:.1f} GB; update the cell and examples/notebooks/README.md together"
    )
