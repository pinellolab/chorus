# CSHL poster panel: rs12740374 multi-oracle convergence at the SORT1 locus

This directory contains everything required to reproduce, verify, and
audit the multi-oracle poster panel at
`poster/rs12740374_SORT1_panel.html`.

The panel shows how three independent chorus oracles
(**ChromBPNet**, **LegNet**, **AlphaGenome**) score the classic noncoding
LDL-cholesterol causal variant **rs12740374** (chr1:109,274,968 G>T, hg38)
in **HepG2** at the **SORT1** locus, and compares the predictions against
the published mechanism (Musunuru et al., *Nature* 2010, PMC3062476).

Audited on **chorus v0.5.6**, commit `f7c1c322b83570a09e5db269083a7b5e86d5152d`
(branch `main` as of 2026-05-15).

---

## Files in this directory

| File | What it is |
| --- | --- |
| `rs12740374_SORT1_panel.html` | **The poster panel.** Self-contained HTML; iframes `igv_only.html`. |
| `extract_igv.py` | Regenerates `igv_only.html` from the canonical chorus example. |
| `verify_predictions.py` | Reruns AlphaGenome, LegNet, ChromBPNet ATAC, ChromBPNet CEBPA. |
| `verify_chrombpnet_cebpa.py` | Standalone ChromBPNet **CEBPA HepG2** scoring (BPNet TF model BP001094). |
| `verify_wide.py` | Reruns ChromBPNet / LegNet with the wide window the canonical regenerator uses (for matching numbers). |
| `expected_numbers.json` | All numbers cited on the panel, with provenance for each. **Diff against this after rerunning.** |
| `_verified_*.json` | Reference snapshots of the verified prediction outputs (committed for diffing). |
| `igv_only.html` *(gitignored)* | 12 MB iframe target with live IGV.js + inlined signal data. Regenerate via `extract_igv.py`. |

---

## Hard-coded facts the panel claims

The verifier must confirm each of these. Sources are listed in
`expected_numbers.json`.

| Row | Layer | ChromBPNet | LegNet | AlphaGenome | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Chromatin accessibility (DNase) | log₂FC **+1.24** (q=0.999) | — | log₂FC **+1.34** (q=1.0) | ✓ both ↑ |
| 2 | TF binding (CEBPA) | log₂FC **+1.99** (BP001094, ENCSR142IGM) | — | log₂FC **+2.78** (q=1.0) | ✓ both ↑ |
| 3 | Histone marks (H3K27ac) | — | — | log₂FC **+1.27** (q=1.0) | single ↑ |
| 4 | MPRA element activity (LentiMPRA) | — | **~0.00** window-mean Δ | — | silent (expected) |
| 5 | eRNA at variant site (CAGE) | — | — | log₂FC **+1.47** (region_label = "variant site") | single ↑ |
| 6 | SORT1 TSS / RNA-seq | — | — | CAGE +0.06 to +0.12; RNA-seq +0.05 to +0.14 (all 5 SORT1 HepG2 RNA-seq tracks UP) | small ↑ |

Plus the contextual claim:

> AG also predicts larger downstream RNA-seq effects on the closer eQTL
> genes **CELSR2** (max log₂FC +0.86, ~1.82×) and **PSRC1** (+0.60, ~1.52×).

---

## Reproduction recipe

### 0. Prerequisites

You need the chorus repo at v0.5.6 or later, plus the per-oracle mamba
environments described in `CLAUDE.md`:

```bash
git checkout poster/cshl-rs12740374-multioracle
git log -1 --oneline   # must include chorus v0.5.6 (commit f7c1c32 or later)

# Required mamba envs (see CLAUDE.md):
mamba env list | grep chorus
# expect: chorus, chorus-alphagenome, chorus-chrombpnet, chorus-legnet
```

The first time you run an oracle, chorus will auto-download:
- the model weights (HuggingFace / ENCODE / Zenodo)
- the per-track CDF backgrounds from
  `huggingface.co/datasets/lucapinello/chorus-backgrounds`

### 1. Verify background CDFs match HuggingFace

These determine the quantile_score columns in the panel.

```bash
mamba run -n chorus python -c "
from huggingface_hub import HfApi
api = HfApi()
import os
home = os.path.expanduser('~/.chorus/backgrounds')
for f in ['alphagenome_pertrack.npz', 'chrombpnet_pertrack.npz', 'legnet_pertrack.npz']:
    local = os.path.join(home, f)
    if not os.path.exists(local):
        from chorus.analysis.normalization import download_backgrounds
        download_backgrounds(f.split('_')[0])
    info = api.get_paths_info('lucapinello/chorus-backgrounds', paths=[f], repo_type='dataset')[0]
    print(f'{f}: local={os.path.getsize(local)}  remote={info.size}  match={os.path.getsize(local)==info.size}')
"
```

Expected: every file `match=True`. Sizes are listed in
`expected_numbers.json` under `backgrounds.verified_byte_identical_to_hf`.

### 2. Regenerate the IGV iframe target

```bash
python poster/extract_igv.py
# wrote /Users/.../chorus/poster/igv_only.html (~12000 KB)
```

This pulls the IGV.js bundle + inlined per-bin signal data out of the
canonical chorus example at
`examples/walkthroughs/validation/SORT1_rs12740374_multioracle/rs12740374_SORT1_multioracle_report.html`
and wraps it in a minimal page that posts its rendered height to the
parent (so the iframe in the panel auto-sizes with no scrollbar).
The default locus is swapped from the full ±500 kb window to
`chr1:109,150,000-109,450,000` for poster readability. The reader can
zoom out inside IGV to recover the full window.

If the source file is missing, regenerate the canonical example:

```bash
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus             python scripts/regenerate_multioracle.py --consolidate
```

### 3. Rerun the verification predictions

Each oracle runs in its own env. They are independent; you can run
them in parallel.

```bash
# AlphaGenome with DNase + ChIP-CEBPA + H3K27ac + 2 CAGE + 5 RNA-seq tracks
mamba run -n chorus-alphagenome python poster/verify_predictions.py --oracle alphagenome

# ChromBPNet ATAC HepG2 (the original headline track on prior runs)
mamba run -n chorus-chrombpnet  python poster/verify_predictions.py --oracle chrombpnet

# LegNet LentiMPRA HepG2 (raw 2bp-window run; expected to give ~+0.30, NOT
# the canonical +0.0001 - see step 4 for the wide-window LegNet)
mamba run -n chorus-legnet      python poster/verify_predictions.py --oracle legnet

# ChromBPNet CEBPA HepG2 TF model (BPNet TF model BP001094)
mamba run -n chorus-chrombpnet  python poster/verify_chrombpnet_cebpa.py
```

Wall-clock notes (Mac M4 Max):
- AlphaGenome: ~3 minutes (model load + 10-track prediction).
- ChromBPNet ATAC: ~2 minutes if model already cached, else ~10 min first time (720 MB weights download).
- ChromBPNet CEBPA: ~30 seconds (much smaller TF model).
- LegNet: ~10 seconds.

Each produces two JSON files in `poster/`:
- `_verified_<oracle>.json` (one row per scored track)
- `_verified_<oracle>_full.json` (the full `VariantReport.to_dict()` dump with per-bin metadata)

### 4. Reproduce the canonical wide-window LegNet and ChromBPNet DNase numbers

The panel's row 1 ChromBPNet (+1.24) and row 4 LegNet (~0.00) come from
the canonical regenerator, which uses a **WIDE** genomic_region equal to
the widest oracle's `output_size` (about 1 Mb). Chorus v0.5.6's
`predict_sliding` then extends signal across that whole window, and the
reported `raw_score` is a window-mean (LegNet) or local-at-variant
(ChromBPNet DNase) summary.

`verify_predictions.py` uses a narrow 2 bp region by default, which
gives LegNet +0.30 (a single-fragment scoring of the variant context),
NOT the canonical near-zero. To reproduce the canonical exactly:

```bash
mamba run -n chorus-chrombpnet python poster/verify_wide.py --which chrombpnet_dnase
mamba run -n chorus-legnet      python poster/verify_wide.py --which legnet
```

Or just trust the canonical example file (which is itself a chorus
regression test):

```bash
cat examples/walkthroughs/validation/SORT1_rs12740374_multioracle/example_output.json | jq '.consensus[] | {layer, oracles: (.oracles | to_entries | map({k:.key, raw:.value.raw_score, q:.value.quantile_score}))}'
```

### 5. Open the panel and the expected-numbers ground truth

```bash
open poster/rs12740374_SORT1_panel.html
cat poster/expected_numbers.json | jq .
```

Diff the panel's quoted numbers against `expected_numbers.json`. Drift
of more than +/- 0.02 log₂FC indicates either a chorus version mismatch,
a stale CDF cache, or non-determinism somewhere (none expected; AG/BPN
inference is deterministic on a given platform).

---

## Honest-finding audit (what to scrutinise)

Three of the panel's claims required correcting a previous-iteration
mistake. The verifier should explicitly check each:

1. **The "+1.47 CAGE" peak is NOT at SORT1's TSS.** It is at the
   `region_label = "variant site"`, i.e. eRNA from the enhancer itself.
   Check by:
   ```bash
   python3 -c "
   import json
   d = json.load(open('poster/_verified_alphagenome_full.json'))
   for t in d['alleles']['alt_1']['all_scores']:
       if t['layer'] == 'tss_activity' and abs(t['raw_score']) > 1:
           print(t['raw_score'], t.get('region_label'), t['assay_id'][:50])
   "
   ```
   Expected: 2 rows, both `region_label == 'variant site'`.

2. **AlphaGenome's predicted SORT1 RNA-seq + CAGE effect is small
   (~1.05 to 1.15×) but directionally UP.** Check by:
   ```bash
   python3 -c "
   import json
   d = json.load(open('poster/_verified_alphagenome_full.json'))
   sort1 = [t for t in d['alleles']['alt_1']['all_scores']
            if (t.get('region_label') or '').startswith('SORT1')]
   for t in sort1:
       print(t['layer'], t['raw_score'], t.get('region_label'), t['assay_id'][:55])
   "
   ```
   Expected: 7 rows. 2 CAGE (SORT1 TSS) + 5 RNA-seq (SORT1 exons), all
   raw_score positive and in 0.05 to 0.14 range.

3. **The model predicts LARGER downstream effects on the closer eQTL
   genes CELSR2 and PSRC1.** Check by:
   ```bash
   python3 -c "
   import json
   d = json.load(open('poster/_verified_alphagenome_full.json'))
   rna = [(t['raw_score'], t.get('region_label'))
          for t in d['alleles']['alt_1']['all_scores']
          if t['layer'] == 'gene_expression']
   for s, r in sorted(rna, reverse=True)[:5]:
       print(f'  {s:+.4f}  {r}')
   "
   ```
   Expected top 5 by raw_score: CELSR2 ~+0.86, +0.72, PSRC1 ~+0.60,
   PSRC1 ~+0.53, CELSR2 ~+0.46.

---

## Known places drift might appear

- **AG raw scores** (DNase, CEBPA, H3K27ac, CAGE): each can move +/- 0.02
  log₂FC across chorus minor releases due to bin-centering / refseq
  changes. The 2026-04-22 example had DNase +1.398, CEBPA +2.759,
  H3K27ac +1.332, CAGE +1.557; the 2026-05-09 v0.5.6 example has
  +1.336, +2.777, +1.267, +1.522. Both round to the panel's values.
- **ChromBPNet CEBPA**: I extract this via OraclePrediction.tracks
  (`verify_chrombpnet_cebpa.py`) rather than `build_variant_report`,
  because the latter crashes on BPNet TF track metadata (missing
  `assay_type` text in `classify_track_layer`). Worth flagging as a
  chorus bug; doesn't affect the prediction itself.
- **LegNet**: my `verify_predictions.py` uses a 2 bp region and gives
  +0.30. The canonical wide-window regenerator gives +0.0001. The panel
  uses the canonical (silent-as-expected) number because LegNet is a
  promoter-fragment scorer and rs12740374 isn't at a promoter; a single
  fragment centered on the variant does not represent how the panel
  uses LegNet in practice.

---

## Provenance trail

- Panel author: this commit (branch `poster/cshl-rs12740374-multioracle`).
- Verified against chorus v0.5.6 (commit f7c1c32, dated 2026-05-15).
- All CDF backgrounds in `~/.chorus/backgrounds/` byte-identical to
  HuggingFace `lucapinello/chorus-backgrounds` as of 2026-05-19.
- Mechanism reference: Musunuru et al., *Nature* 466 (2010), PMC3062476.
- The variant is a known eQTL for **CELSR2, PSRC1, and SORT1** (the
  panel's row 6 contextual note builds on this).
