"""``device='cuda:N'`` means the Nth GPU *this process can see* (audit F1).

Four oracle templates used to do this:

    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]

which is wrong in two different ways depending on the framework, and both were live.

**Under a scheduler it steals a GPU.** Given ``CUDA_VISIBLE_DEVICES=4,5`` — the two devices a
queue granted this job — a request for ``cuda:1`` means "the second one I was given", physical
5. Overwriting the mask with ``1`` puts the process on **physical GPU 1**, which belongs to
somebody else. Not hypothetical on the host this was found on: GPU 4 was another tenant's
throughout the audit.

**For the torch oracles it also crashed.** Cherimoya's templates set the mask to ``N`` and then
hand the *same* string ``cuda:N`` to torch, which indexes within the now-1-device visible set —
so every ordinal except 0 died with ``CUDA error: invalid device ordinal``. Verified directly:
masking to one device and calling ``.to('cuda:1')`` raises, while ``.to('cuda:0')`` works and
lands on physical GPU 1. So the documented ``device='cuda:N'`` parameter was broken for
Cherimoya, not merely unsafe.

The fixes differ because the selection mechanism differs, and that is the point worth keeping:

* **torch** (cherimoya) — do not touch the variable at all. Torch already interprets the
  ordinal within the visible set, so the mask assignment was both redundant and harmful.
* **TensorFlow** (chrombpnet) — the mask *is* the selection mechanism, so ``cuda:N`` is
  remapped through whatever mask is already set, and an out-of-range ordinal raises with a
  message that says what N indexes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = sorted(REPO.glob("chorus/oracles/*_source/templates/*template.py"))

#: The exact shape of the defect: assigning the mask straight from the ordinal.
UNSAFE = re.compile(
    r"CUDA_VISIBLE_DEVICES'?\"?\]\s*=\s*(?:device\.split|gpu_id\s*$|str\(int\()")


def test_there_are_templates_to_check():
    assert len(TEMPLATES) >= 4, f"expected the oracle templates, found {TEMPLATES}"


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_no_template_overwrites_the_mask_with_a_bare_ordinal(path: Path):
    """The enumeration guard. Four templates had this; a fifth must not reintroduce it."""
    offenders = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if "CUDA_VISIBLE_DEVICES" not in code or "=" not in code:
            continue
        # Forcing CPU with -1 is correct and stays.
        if "'-1'" in code or '"-1"' in code:
            continue
        # Remapping through the existing mask is the fix, not the defect.
        if "visible[" in code or "gpu_id" in code and "visible" in path.read_text():
            continue
        if UNSAFE.search(code) or "device.split" in code:
            offenders.append(f"{path.name}:{i}  {line.strip()[:78]}")
    assert not offenders, (
        "these templates set CUDA_VISIBLE_DEVICES from a bare ordinal, which overrides an "
        f"outer scheduler mask and lands on a GPU the caller was not granted:\n  "
        + "\n  ".join(offenders)
    )


def test_the_torch_templates_leave_the_mask_alone_for_cuda_n():
    """Torch resolves the ordinal itself; touching the mask made cuda:1 crash."""
    for name in ("load_template.py", "predict_template.py"):
        src = (REPO / "chorus" / "oracles" / "cherimoya_source" / "templates" / name).read_text()
        assert "device.split" not in src, (
            f"cherimoya/{name} is masking again; with the mask set to N, the resolved "
            f"'cuda:N' handed to torch is out of range for every N but 0"
        )
        # Forcing CPU must still work.
        assert '"-1"' in src or "'-1'" in src


def test_the_tensorflow_templates_remap_through_the_existing_mask():
    """TF selects via the mask, so cuda:N must be translated, not substituted."""
    for name in ("load_template.py", "predict_template.py"):
        src = (REPO / "chorus" / "oracles" / "chrombpnet_source" / "templates" / name).read_text()
        assert "visible[int(ordinal)]" in src, (
            f"chrombpnet/{name} no longer remaps cuda:N through CUDA_VISIBLE_DEVICES"
        )
        assert "indexes the devices you were granted" in src, (
            "an out-of-range ordinal should say what N indexes, since the whole confusion "
            "is physical-vs-visible"
        )


def test_the_remap_arithmetic_is_right():
    """Executed, not just grepped: lift the logic and check it against a real mask.

    ``cuda:1`` under a ``4,5`` grant must resolve to physical **5**, and ``cuda:2`` must raise
    rather than silently pick something.
    """
    def remap(device: str, outer: str | None):
        ordinal = device.split(":")[1]
        if outer:
            visible = [x for x in outer.split(",") if x != ""]
            try:
                return visible[int(ordinal)]
            except (ValueError, IndexError):
                raise ValueError(f"cuda:{ordinal} against {outer!r}")
        return ordinal

    assert remap("cuda:0", "4,5") == "4"
    assert remap("cuda:1", "4,5") == "5"
    assert remap("cuda:3", None) == "3"          # no mask: pass through, as before
    with pytest.raises(ValueError):
        remap("cuda:2", "4,5")

    # And the same arithmetic must be what the template actually contains.
    src = (REPO / "chorus" / "oracles" / "chrombpnet_source" / "templates"
           / "predict_template.py").read_text()
    assert "[x for x in outer.split(',') if x != '']" in src or \
           "[p for p in outer.split(',') if p != '']" in src

# ──────────────────────────────────────────────────────────────────────
# The direct path — the half the first fix missed
# ──────────────────────────────────────────────────────────────────────

#: Oracle modules, not templates. `use_environment=False` is the `create_oracle` DEFAULT, so
#: this path matters at least as much as the subprocess one — and it was still overwriting the
#: mask after the templates were fixed, because the original guard only globbed
#: `*_source/templates/*template.py`. Found by re-auditing the released tree (2026-08-13).
ORACLE_MODULES = sorted(REPO.glob("chorus/oracles/*.py"))


@pytest.mark.parametrize("path", ORACLE_MODULES, ids=lambda p: p.name)
def test_no_oracle_module_overwrites_the_mask_with_a_bare_ordinal(path: Path):
    """Same property as the template guard, on the in-process load path."""
    offenders = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        code = line.split("#", 1)[0]
        if "CUDA_VISIBLE_DEVICES" not in code or "=" not in code:
            continue
        if "'-1'" in code or '"-1"' in code:          # forcing CPU is correct
            continue
        # The value assigned must come from the resolver, not from splitting the string.
        window = "\n".join(lines[max(0, i - 6):i])
        if "resolve_visible_ordinal" in window:
            continue
        if "device.split" in code or "device.split" in window:
            offenders.append(f"{path.name}:{i}  {line.strip()[:78]}")
    assert not offenders, (
        "these oracle modules set CUDA_VISIBLE_DEVICES from a bare ordinal on the direct "
        f"load path, so cuda:N lands on physical N regardless of the mask:\n  "
        + "\n  ".join(offenders)
        + "\n  Use chorus.core.platform.resolve_visible_ordinal()."
    )


def test_the_shared_resolver_is_the_only_arithmetic():
    """One implementation now, after six templates and two modules had their own."""
    from chorus.core.platform import resolve_visible_ordinal
    import os

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
        assert resolve_visible_ordinal("cuda:0") == "4"
        assert resolve_visible_ordinal("cuda:1") == "5"
        with pytest.raises(ValueError, match="granted"):
            resolve_visible_ordinal("cuda:2")
        del os.environ["CUDA_VISIBLE_DEVICES"]
        assert resolve_visible_ordinal("cuda:3") == "3", "no mask: pass the ordinal through"
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
