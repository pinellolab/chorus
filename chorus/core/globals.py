from pathlib import Path
CHORUS_ROOT = Path(__file__).parent.parent.parent

CHORUS_ANNOTATIONS_DIR = CHORUS_ROOT / "annotations"
CHORUS_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

CHORUS_DOWNLOADS_DIR = CHORUS_ROOT / "downloads"
CHORUS_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)