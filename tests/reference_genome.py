"""Where the suite finds the hg38 reference: ``<data dir>/genomes/hg38.fa``.

Not a test module (no ``test_`` prefix, so pytest does not collect it). Each test that needed the
genome used to say where it was, and nine files said ``genomes/hg38.fa`` relative to the repo or
the cwd. That was right only while the data dir was always the install dir. Bulk data now follows
``CHORUS_DATA_DIR`` and ``chorus_data_dir.txt``, so a host set up the documented way keeps the genome
elsewhere. Those tests then skipped as though it were missing, and two with no guard failed with
``OSError``. The v0.8.0 audit (P1-14, 2026-11-15) counted 23 integration skips and 22 fast-suite
skips from this alone, among them all six oracle smoke tests: a green ``pytest -m integration``
that had verified none of them.

Resolved through ``chorus.core.globals``, so the suite looks exactly where
``chorus genome download hg38`` writes and where the MCP server reads.
"""
from __future__ import annotations

from chorus.core.globals import CHORUS_GENOMES_DIR

#: The reference FASTA, wherever this install keeps its data.
HG38_FASTA = CHORUS_GENOMES_DIR / "hg38.fa"

#: Its index, which is all the region samplers read (chromosome sizes).
HG38_FAI = CHORUS_GENOMES_DIR / "hg38.fa.fai"
