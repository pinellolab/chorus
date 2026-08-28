"""A custom annotation's HF download must be pinned to a real revision.

tests/test_artefact_revision_is_pinned.py enforces this for chorus/analysis/
normalization.py's background artefacts, via a fixed `_HF_REVISION` constant. This
module has no such constant — the revision is supplied per-annotation through
AnnotationStore.add_annotation's required `hf_revision` argument (rejecting
main/master/HEAD is covered in tests/test_annotation_store.py) — but the same
"one unpinned call is enough to reintroduce the drift" risk applies to the actual
hf_hub_download call site, so it gets the same source-scan check here rather than
silently relying on the existing test only scanning normalization.py.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANNOTATION_STORE = REPO / "chorus" / "utils" / "annotation_store.py"


def test_every_hf_hub_download_call_site_passes_revision():
    src = ANNOTATION_STORE.read_text()
    positions = [m.start() for m in re.finditer(r"hf_hub_download\(", src)]
    assert positions, "no hf_hub_download call found -- has the download path moved?"

    unpinned = []
    for pos in positions:
        call = src[pos:pos + 400]
        end = call.find(")")
        window = call[: end if end > 0 else len(call)]
        if "revision=" not in window:
            line = src[:pos].count("\n") + 1
            unpinned.append(line)

    assert not unpinned, (
        f"hf_hub_download at line(s) {unpinned} in {ANNOTATION_STORE} does not pass "
        f"revision=, so it would fetch the repo's moving head regardless of what the "
        f"caller declared."
    )
