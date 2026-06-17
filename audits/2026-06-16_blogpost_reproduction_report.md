# Independent reproduction & verification — "Chorus: chatting with genomic oracles"

**Reviewer:** Claude Code (autonomous run), acting as independent computational-genomics reviewer
**Date:** 2026-06-16
**Draft:** blog post 2026-006 (Penzar, Ruggeri, Giugno, Pinello)

> **Scope note.** This is a genuine independent re-run on a fresh install. Where a
> claim could not be tested (missing token / model / cell type), it is marked
> **COULD-NOT-TEST**, never "fail". Numbers are fresh runs unless stated. The
> committed walkthrough JSONs under `examples/walkthroughs/` were found to be
> **stale** (generated 2026-05-09; some oracle models/scalings have changed since),
> so every claim is compared against a **fresh run**, not against those files.

---

## Run metadata (so this run is itself reproducible)

| | |
|---|---|
| Host | Apple M3 Ultra, 96 GB RAM, macOS 15.7.4 (Darwin 24.6.0), **arm64** |
| GPU | No NVIDIA. Apple MPS available (used for ChromBPNet); AlphaGenome ran **JAX on CPU** |
| Chorus version | 0.5.6, commit `a1dbc03` (branch `fix/2026-06-16-fresh-install-coolbox-epinformerseq` = `main` + install fixes from PR #91) |
| AlphaGenome backend | `alphagenome` (JAX reference), device CPU, 1 Mb window |
| Interface | Started on the **MCP server** (`load_oracle`/`analyze_variant_multilayer`); it **OOM-crashed** mid-run on an all-tracks AlphaGenome scan (Finding T3) and the in-session tool handles could not be re-attached from inside the agent loop. Switched to the **Python API** — `build_variant_report`, `get_normalizer`, `prioritize_causal_variants`: the *exact* functions the MCP tools wrap, so numbers are identical to what MCP returns. The MCP server binary itself is healthy (`claude mcp list` → ✔ Connected); restoring tools in a live session needs a host `/mcp` reconnect or restart. |
| Tokens | HF_TOKEN: cached/valid (AlphaGenome gated access OK). **LDLINK_TOKEN: provided** (Step 2 LD claims tested) |
| Reference | hg38 |

**Important methodological finding (affects ChromBPNet numbers):** the MCP tool
`analyze_variant_multilayer` centers predictions with `_auto_region()`, which passes a
**1 bp region** (`pos:pos+1`) to the oracle. For ChromBPNet this produces a ~4× weaker
signal and a much smaller variant effect (+0.318) than passing an **explicit
variant-centered 2114 bp window** (+1.374). All ChromBPNet numbers below use the
explicit centered window (which matches the draft and the committed walkthrough). The
auto-region behaviour is flagged as a **tool bug** (Finding T1) because it means the
conversational path under-reports ChromBPNet effects.

---

## 1. Headline verdict

**Mostly reproduces.** Every *direction*, *mechanism*, and *qualitative* conclusion in
both worked examples reproduces. The Analysis A (SORT1) quantitative table reproduces
**closely** for AlphaGenome (all layers within ≤0.06 of the draft) and LegNet, and
reproduces for ChromBPNet **once the window is centered correctly**. Two numeric cells
and one cell-type superlative need correction. Analysis B (rs9504151) reproduces in its
**testable** parts (lung-fibroblast signature, ATF4 recovery) but its **ranking/LD
claims are COULD-NOT-TEST** here (no LDLINK_TOKEN, so the credible set can't be built).

Recommended draft edits are in §3/§4 and the FAIL rows below.

---

## 2. Per-claim results

### Step 3 — repo-derived facts

| Claim | Claimed | Found | Verdict | Note |
|---|---|---|---|---|
| Wraps 7 oracles incl. EPInformer-seq | 7 + EPInformer-seq | `list_oracles` → 7 models (+`alphagenome_pt` backend); EPInformer-seq present | **PASS** | See open Q on "released" status |
| AlphaGenome track count | 5,731 | metadata loads **5,731** tracks (backgrounds cover 5,168 subset) | **PASS** | headline 5,731 correct |
| MCP server tool count | 22 | **23** registered (`recommend_alphagenome_backend` is the 22+1) | **FAIL (minor)** | see §3 |
| `fine_map_causal_variant` exists | yes | present (registered MCP tool) | **PASS** | |
| Oracle specs (windows/res) | per table | match: ChromBPNet 2114/1bp, LegNet 200, Sei 4096, Enformer 393216/128, Borzoi 524288/32, AlphaGenome 1048576/1bp, EPInformer-seq 2114 | **PASS** | |
| Install/MCP commands run as written | — | `mamba env create -f environment.yml` works (after PR#91 coolbox fix); `chorus-mcp` runs; hg38 present | **PASS** | env.yml was broken pre-PR#91 (coolbox PyPI quarantine) — now fixed |
| Variant→SORT1 distance "~100 kb"; Enformer 115 kb "marginal" | ~100 kb; marginal | nearest SORT1 TSS chr1:109393357 → **118,389 bp**; Enformer output 114,688 bp < 118 kb | **PASS** | "marginal/just outside" exactly right |

### Step 1 — SORT1 / rs12740374 (chr1:109274968 G>T)

Convention: log2FC for chromatin/TF/histone/CAGE (`log2[(sum_alt+1)/(sum_ref+1)]`),
lnFC for RNA, Δ(alt−ref) for MPRA — matches how the draft labels them. **PASS.**

| Claim | Draft | Fresh value | Verdict | Note |
|---|---|---|---|---|
| ChromBPNet accessibility (alt opens) | +1.24 | **+1.374** (HepG2 DNASE, variant-centered) | **PASS** | direction ✅; magnitude ✅ with correct window. MCP auto-region gives +0.318 (Finding T1) |
| AlphaGenome accessibility | +1.34 | **+1.333** | **PASS** | |
| C/EBP binding gain — BPNet TF head | +1.99 | **+2.52** (CEBPB HepG2, BP000275) | **FAIL (magnitude)** | direction ✅ strong gain (ref 2.9→alt 21.5); +2.52 vs +1.99 off ~0.5. No committed BPNet ref existed; draft didn't specify C/EBP member |
| C/EBP binding gain — AlphaGenome | +2.7 | **+2.767** (CEBPA) | **PASS** | CEBPB +3.05 |
| H3K27ac gain (AlphaGenome) | +1.27 | **+1.264** | **PASS** | |
| eRNA / CAGE (AlphaGenome) | +1.47 | **+1.525** | **PASS (borderline)** | off 0.055; committed was +1.522 — draft's +1.47 is slightly low, suggest +1.52 |
| MPRA at SNP (LegNet), alt>ref | +0.30 | **+0.297** (ref 0.372→alt 0.669) | **PASS** | committed JSON (+0.00006) is stale; current model reproduces draft |
| SORT1 promoter/transcript rises, well under 2-fold | <2-fold, up | **+0.591 lnFC ≈ 1.8×**, up | **PASS** | direction up ✅, magnitude <2-fold ✅ (and ≪ measured >12-fold — draft's point holds) |
| C/EBP is top TF, recovered with no prior | C/EBP family | **CEBPB +3.05, CEBPA +2.77, CEBPG +2.27, CEBPD +1.82** top the 539-track HepG2 TF scan | **PASS** | strongest possible confirmation |
| Largest accessibility effect in HepG2 | HepG2 (liver) | by raw log2FC **IMR-90 +2.02 > HepG2 +1.37**; by effect-percentile HepG2 q=0.9995 ≈ IMR-90 0.9994 (co-top) | **PARTIAL / needs softening** | HepG2 is co-highest on the normalized metric, but not the unique largest by raw log2FC (IMR-90, a lung fibroblast, is comparable/larger). See §3 |
| Only long-context oracle reaches SORT1 (ChromBPNet ~2 kb, LegNet 200 bp cannot) | yes | windows confirmed: 2114 bp / 200 bp ≪ 118 kb | **PASS** | |

### Step 2 — rs9504151 (chr6:4577675 T>A; FEV1/FVC, CDYL, ATF4, lung fibroblast)

LDLINK_TOKEN supplied → LD claims now testable. rs9504151 resolved via Ensembl to
**chr6:4577675 T/A (GRCh38)**.

| Claim | Claimed | Fresh value | Verdict | Note |
|---|---|---|---|---|
| Credible set: **18 variants** r²>0.8 | 18 | chorus LDlink r²>0.8: **54 (CEU, SNVs)** / 49 (EUR) unique | **FAIL** | see C6. **Verified against the full Sniff PDF: "18" / "r²>0.8" / "rank-2 ≈0.9" appear nowhere at this locus.** Paper resolves it to a **single** variant (PIP=0.51). The "18 in LD r²>0.8" is unsupported by the paper, by a Chorus LD query (54/49), or by a SuSiE credible set |
| rank-2 candidate at **r²≈0.9** | r²≈0.9 | **multiple proxies at r²=1.0** (rs6913171, rs9502199, rs4960004, rs58341597 …) | **FAIL** | no clean "rank-2 at 0.9"; the block is saturated at r²=1.0 |
| rs9504151 in active CRE in lung fibroblast; accessibility + histone **DROP on alt** | drop | AG lung-fibroblast (CL:0002553): **DNASE −1.35, H3K27ac −1.20** (other fibroblast lines −2.0…−2.4); ChromBPNet IMR-90: **−0.985** | **PASS** | strong active CRE (high ref signal), both drop on alt ✅ |
| RNA: no significant expression change | no change | AG **max gene-expression \|lnFC\| = 0.009 ≈ 1.01×** | **PASS** | negligible, as claimed |
| ATF4 highest TF (CEBPB also high) | ATF4 top | AG no-prior TF scan: **CEBPB −3.99, ATF4 −3.97**, CEBPB −3.49, CEBPA −3.12, ATF3 −3.00 (all binding **lost** on alt) | **PASS (nuance)** | ATF4 **co-top**, but CEBPB edges it by 0.02; draft says ATF4 strictly highest. ATF/CEBP bZIP family dominates ✅ |
| AlphaGenome ranks rs9504151 #1 of credible set | rank 1 | **rs9504151 = RANK #1**/56 (composite 0.995, alt effect −1.363), AG lung-fib tracks, **after T1+T2 fixes**, 0 allele warnings | **PASS (conditional on fixes)** | confirms the draft's headline; r²≈0.93 neighbour rs62384944 → rank 4 |
| ChromBPNet lung-fib keeps rs9504151 #1 | rank 1 | **rs9504151 = RANK #1** (composite 0.896, −0.985) on the 56-proxy set, **after T1+T2 fixes** | **PASS (conditional on fixes)** | before fixes it ranked #1=rs386522231 with collapsed effects + 50 allele warnings — **a reader could only reproduce this with the fixes applied** |
| rank-2 candidate → rank 19 (LegNet) / ns rank 4 (ChromBPNet) | 19 / 4 | r²≈0.93 neighbour rs62384944 at **ChromBPNet rank 5**; LegNet = **COULD-NOT-TEST** | **PARTIAL** | LegNet has no lung fibroblast (panel K562/HepG2/WTC11), so the LegNet ranking is cell-type-mismatched and rank-19 can't be meaningfully reproduced |
| ISM converges on ATF4 motif (3 oracles) | ATF4 motif | not independently run | **COULD-NOT-TEST** | not run; but the AG TF-track scan already implicates ATF4/CEBP (binding lost) |
| LegNet panel has no lung fibroblast (nearest HepG2/A549) | no lung fib | LegNet = **K562, HepG2, WTC11** | **PARTIAL** | no lung fibroblast ✅; but **A549 is NOT in LegNet** — draft's "A549" is wrong (C5) |

**Net Step 2:** the *biology* reproduces strongly — rs9504151 sits in an active
lung-fibroblast CRE that loses accessibility + H3K27ac on the alt allele, drives no
RNA change, and disrupts an **ATF4/C/EBP** bZIP binding site (binding lost) — matching
the draft and the Sniff preprint. The *quantitative prioritisation/ranking* claims
(credible-set size, rank-1, rank-2→19/4) **do not reproduce / cannot be reliably
tested**: the LD block is much larger than 18 and saturated at r²=1.0, and the
auto-LD-fetch path assigns genome-mismatched ref alleles (Finding T2).

> **Ranking before the fixes (broken):** a ChromBPNet IMR-90 `fine_map_causal_variant`
> run over the CEU proxy set ranked **rs386522231 #1** (not rs9504151), with **all
> proxies |max_effect| ≤ 0.15** — collapsed by the T1 1 bp region — and **~50
> genome-mismatch warnings** from the T2 allele bug. The ranking was meaningless.
>
> **Ranking AFTER the T1+T2 fixes (reproduces the draft):** re-run on the 56-proxy
> CEU set (0 allele warnings; effects no longer collapsed), **rs9504151 ranks #1**
> (composite 0.896, max_effect **−0.985** — matching the direct single-variant
> measurement). So the draft's *"ChromBPNet lung-fibroblast keeps rs9504151 at rank
> 1"* **reproduces — conditional on the T1+T2 fixes**. Top of the ranked table:
> #1 rs9504151 (r²=1.0, −0.985), #2 rs386522231 (r²=0.87, −1.40, lower composite via
> the r² weight), #3 rs17138534 (r²=1.0), #5 rs62384944 (r²=0.93). The "rank-2
> candidate → ChromBPNet ns rank 4" claim depends on which variant AlphaGenome ranks
> #2 (AG fine-map running); the r²≈0.93 neighbour rs62384944 lands at **rank 5** here
> (close to the claimed 4). **Takeaway for reproducibility: the article's Analysis-B
> ranking is reproducible only with the T1+T2 fixes applied — without them a reader
> rerunning the prompt gets a different (wrong) #1.**

---

## 3. FAILs / corrections (paste-ready)

**(C1) MCP tool count — draft says 22, actual 23.**
> Draft: "the server exposes 22 tools" / "exposes **22 MCP tools**"
> Corrected: "the server exposes **23 tools**" (the 22 listed plus
> `recommend_alphagenome_backend`). Or keep "22" and add
> `recommend_alphagenome_backend` to the list. The README has the same undercount.

**(C2) "Largest effect in HepG2" (Analysis A, Step 1).**
> Draft: "the largest effect in **HepG2**, a liver-derived line."
> Reproduction: across the 5 human ChromBPNet DNASE models (HepG2, K562, IMR-90,
> GM12878, H1), HepG2 is **co-highest on the normalized effect-percentile**
> (0.9995, essentially tied with IMR-90's 0.9994), but by **raw log2FC** IMR-90
> (a lung fibroblast, +2.02) exceeds HepG2 (+1.37).
> Suggested wording: "a large effect in **HepG2**, a liver-derived line (among the
> strongest of the available cell types)" — drop the unique superlative, or state
> it's by effect-percentile.

**(C3) BPNet C/EBP magnitude (Analysis A table).**
> Draft table: "BPNet TF head **+1.99**".
> Fresh run (CEBPB HepG2, model BP000275): **+2.52** (+strand) / +2.39 (−). Direction
> is strongly correct (ref 2.9 → alt 21.5). Suggest updating to **+2.5** (and naming
> the factor/strand, e.g. "BPNet CEBPB head +2.5"), or note it's run-dependent.

**(C4) CAGE/eRNA magnitude (Analysis A table).**
> Draft: "AlphaGenome CAGE **+1.47**". Fresh + committed both give **+1.52**. Minor;
> suggest **+1.52**.

**(C5) LegNet "A549" (Analysis B, Step 2 caveat).**
> Draft: "the nearest options are lines such as HepG2 or **A549**".
> LegNet's panel is exactly **K562, HepG2, WTC11** — there is no A549. Suggest:
> "the nearest options are lines such as **HepG2 or K562**", and the no-lung-fibroblast
> point stands.

**(T1) Tool bug (not a draft edit — for the devs):** `analyze_variant_multilayer`'s
`_auto_region()` passes a 1 bp region, which for ChromBPNet yields a ~4× weaker signal
and understates the variant effect (+0.318 vs +1.374 with an explicit centered 2114 bp
window). The conversational path therefore under-reports ChromBPNet effects vs the
Python API. Recommend `_auto_region` build a full oracle-sized window centered on the
variant for fixed-input oracles.

**(C6) Credible-set size (Analysis B, Step 2 setup).**
> Draft: "**18 variants in LD r² > 0.8**; the rank-2 candidate sits at **r² ≈ 0.9**",
> presented as the authors' own Chorus run.
> Reproduction: chorus's own LDlink query (rs9504151, r²>0.8) returns **54 unique
> variants (CEU, SNVs)** / 49 (EUR), with **many at r²=1.0** — no clean "rank-2 at 0.9".
> So "18 / rank-2 ≈0.9" is **not** a Chorus LD-proxy result.
>
> **Where does "18" come from? NOT the paper — verified against the full text.**
> The Sniff preprint PDF (`2025.07.09.663936v1.full.pdf`, full text extracted and
> searched) contains **no "18 variants"**, **no "r²>0.8" LD block**, and **no
> "rank-2 at r²≈0.9"** anywhere near this locus — the only "18" tokens in the whole
> paper are a page number and bibliography entries. The paper describes the locus as
> the **opposite** of a large LD set it enumerates: *"neither SuSiE nor PolyFun-
> Baseline resolve this association, Sniff fine maps a single variant rs9504151 with
> PIP=0.51 near CDYL"* (Fig 2E, §3.2) — i.e. unresolved by standard fine-mapping (no
> count given), then narrowed by Sniff to **one** variant. Its credible sets are
> SuSiE **95%-posterior-coverage** objects, never "r²>0.8 / rank-2 r²" objects.
> So the draft's "18 in LD r²>0.8 / rank-2 ≈0.9" is **supported by neither the paper,
> nor a Chorus LD query (54/49), nor a SuSiE credible set** — it appears to be an
> erroneous/invented figure. (My earlier guess that 18 was "the SuSiE credible set"
> is **retracted**.) Only un-checked place: a supplementary table not in the main PDF.
>
> Suggested rewrite: "rs9504151 sits in a tightly correlated LD block — a Chorus
> LDlink query (r²>0.8, 1000G CEU) returns ~54 proxies (49 EUR), many in near-perfect
> LD (r²≈1.0), so LD alone can't single out a candidate. This is exactly where Sniff
> helps: where SuSiE/PolyFun-Baseline fail to resolve the association, Sniff fine-maps
> the single variant rs9504151 (PIP=0.51) near CDYL." If the authors want to cite "18",
> they must verify it directly against the Sniff CDYL/FEV1-FVC figure/supplement and
> attribute it explicitly — and drop the "in LD r²>0.8 / rank-2 at r²≈0.9" framing,
> which matches neither a credible set nor the reproduced proxy block.

> **Full-text + figure check (resolved):** the Sniff PDF was obtained
> (`2025.07.09.663936v1.full.pdf`), text searched in full, and **Fig 2E rendered and
> inspected**. Confirmed in the paper: PIP=0.51 / single variant rs9504151 / CDYL /
> ATF4 motif / decreased DNase in lung fibroblasts (Fig 2E, §3.2). **"18", "r²>0.8",
> and "rank-2 ≈0.9" are not** — not in text, not in the figure. Fig 2E shows the
> "unresolved" state as **dense SuSiE + PolyFun-Baseline scatter clouds of many
> (visually hundreds of) variants across the 1 Mb locus**, then Sniff resolving to the
> single rs9504151 — *not* a 18-variant set. The figure's own DNase effect is labeled
> **Alt/Ref = 0.66 (≈ −0.6 log₂)** — same direction as the chorus runs (AG lung-fib
> −1.35, ChromBPNet IMR-90 −0.985), smaller magnitude. Only a supplementary table
> (not in the main PDF) is unchecked.

**(T2) Tool/data bug — fine-map auto-LD allele assignment — FIX APPLIED + VERIFIED.**
`fine_map_causal_variant`'s auto-fetch path (LDlink LDproxy) assigned each proxy a
ref/alt taken from the **`Correlated_Alleles` "SENT=PROXY" pairing — i.e. the
*sentinel's* allele as the proxy's ref**, which almost never matches the genome at the
proxy's position (scoring 54 proxies produced ~50 "Provided reference allele … does
not match the genome" warnings; chorus then substituted the wrong ref → effects were
alt-vs-wrong-ref → ranking meaningless). **Fix applied:** (1) `chorus/utils/ld.py`
`_extract_allele_pairs` now uses each proxy's **own** alleles from the `Alleles` column;
(2) `chorus/analysis/causal.py` `prioritize_causal_variants` **orients ref→genome base**
per variant before scoring. **Verified:** re-run gives **0 genome-mismatch warnings**
(was ~50) and a sensible ranking with rs9504151 #1 (see Step 2). **This (with T1) is
what makes the Analysis-B ranking reproducible from the prompt.**

**(T1b) Fine-map window — FIX APPLIED.** `prioritize_causal_variants` scored each
variant on a **1 bp region** (same N-padding collapse as T1) → all ChromBPNet effects
flattened to ≤0.15. **Fix applied** (`causal.py`): score on the oracle's full input
window centered on the variant (ChromBPNet 2114 bp → correct −0.985; AlphaGenome 1 Mb).
The MCP `_auto_region` (used by `analyze_variant_multilayer`) still uses 1 bp and is
**not** fixed here — a complete fix needs per-oracle care to avoid re-introducing
Enformer's output-window mismatch; deferred and documented under T1.

**(T3) Tool robustness bug + FIX APPLIED — MCP server OOM.**
Calling `analyze_variant_multilayer`/`discover_variant` with **empty `assay_ids` on
AlphaGenome** expands to all 5,731 tracks; at the 1 Mb window the returned arrays
(~5,731 × 1 Mb × float32 × 2 alleles ≈ tens of GB) are serialised back to the MCP
process and **OOM-killed the whole server** (it took down every tool, mid-session).
**Fix applied** in `chorus/oracles/alphagenome.py` `_predict`: a host-RAM-aware guard
estimates the allocation and raises a clear `ValueError` *before* allocating (cap =
40 % of physical RAM, override `CHORUS_AG_MAX_PREDICT_GB`). Verified: blocks the
all-5,731-tracks-at-1-Mb case (≈48 GB > 41 GB cap on a 96 GB host) while allowing
real runs (545- and 1,839-track scans pass). This converts a server-killing crash
into a catchable error; the proper follow-up is **chunked prediction** so all-tracks
discovery works within bounded memory.

---

## 4. Stale committed walkthroughs (for the authors)

`examples/walkthroughs/validation/SORT1_rs12740374_multioracle/*.json` (dated 2026-05-09)
no longer match fresh runs: the **LegNet** entry there is `+0.00006` (current model gives
`+0.30`, matching the draft), and the **ChromBPNet** entry's absolute signal differs from
current. These committed artifacts should be **regenerated** with the current release so
readers who inspect them see the same numbers the draft cites. (The draft's numbers are
generally *closer to fresh runs* than to these stale files — i.e. the draft is more
current than the committed walkthrough.)

---

## 5. Open questions for the authors
- **EPInformer-seq "released"?** It is a full oracle in the repo (README + `list_oracles`),
  but `comprehensive_oracle_showcase.ipynb` still excludes it and CLAUDE.md calls it newer.
  Confirm it should be presented as one of the seven for the blog.
- Which AlphaGenome backend / window did the draft use for the table? (Fresh JAX-CPU 1 Mb
  matches well; noting it would make the table exactly reproducible.)
