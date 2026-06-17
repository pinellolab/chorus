# Reproduction working notes — Chorus blog post (2026-06-16, autonomous run)

Run metadata: M3 Ultra, 96 GB, arm64, no NVIDIA; chorus 0.5.6 @ commit a1dbc03
(branch fix/2026-06-16-fresh-install-coolbox-epinformerseq = main + install fixes).
Interface: mix of MCP tools (analyze_variant_multilayer etc.) and Python API. AlphaGenome
backend: TBD (JAX CPU for 1 Mb; PT-MPS ≤600 kb).

## Step 3 facts (deterministic)
- 7 oracles incl EPInformer-seq: ✅ list_oracles returns 8 entries = 7 models + alphagenome_pt backend.
- MCP tools: server registers **23** tools (22 documented + recommend_alphagenome_backend). Draft/README say 22 → minor undercount by 1. fine_map_causal_variant present ✅.
- SORT1 TSS nearest = chr1:109393357 (− strand); variant chr1:109274968 → **118,389 bp** (next TSS 122,950). Draft "on the order of 100 kb" ✅. Enformer output window 114,688 bp < 118 kb → "marginal/just outside" ✅.
- Oracle specs (windows/res) from list_oracles match draft table: enformer 393216/128, borzoi 524288/32, chrombpnet 2114/1, sei 4096, legnet 200, alphagenome 1048576/1, epinformerseq 2114. ✅

## Step 1 — committed walkthrough numbers (source of draft table)
examples/walkthroughs/validation/SORT1_rs12740374_multioracle (generated 2026-05-09):
- ChromBPNet DNASE:HepG2 log2FC = **+1.241** (ref 255.66 → alt 605.57)
- LegNet LentiMPRA:HepG2 Δ = **+0.00006** (ref -0.73866 → alt -0.73860)  ← draft says +0.30 ⚠️
- AlphaGenome CEBPA:HepG2 ChIP-TF = **+2.777**  (draft +2.7 ✅)
- AlphaGenome H3K27ac:HepG2 = **+1.267** (draft +1.27 ✅)
- AlphaGenome CAGE:HepG2 = **+1.522** (draft says +1.47 ⚠️ off 0.05)
Scoring conventions (chorus/analysis/scorers.py): chromatin/TF/CAGE = log2[(sum_alt+1)/(sum_ref+1)] over 501 bp; histone 2001 bp; MPRA = simple alt−ref diff; RNA = lnFC mean over exons.

## Step 1 — FRESH reproduction (in progress)
- ChromBPNet DNASE:HepG2, default model (chrombpnet_nobias), **device MPS**: log2FC = **+0.318** (ref 48.5 → alt 60.7)
  ⚠️ Does NOT match committed +1.241. Absolute signal 5× smaller. Hypothesis: MPS (tensorflow-metal) numerics differ from CPU. Testing CPU next.

## Discrepancies flagged so far
1. LegNet MPRA: draft +0.30 vs committed/tool +0.00006 — investigate.
2. CAGE: draft +1.47 vs committed +1.522.
3. ChromBPNet fresh MPS +0.318 vs committed +1.241 — device/model issue under investigation.
4. MCP tools: 23 actual vs 22 claimed.

## KEY INSIGHT (2026-06-16 run): committed walkthrough JSONs are STALE
Models/scaling changed since 2026-05-09. Must compare draft vs FRESH runs, not committed JSONs.

### Step 1 fresh results (CPU unless noted), draft claim → fresh value:
- ChromBPNet DNASE:HepG2 accessibility: draft +1.24 → **fresh +0.318** (nobias default; bias-aware +0.155). Direction ✅ up. Magnitude ❌ (committed old model gave +1.241; current model ~5x lower signal). 
- LegNet LentiMPRA:HepG2 MPRA: draft +0.30 → **fresh +0.297** (ref 0.372→alt 0.669) ✅ PASS (alt>ref). NOTE committed JSON +0.00006 is stale.
- AlphaGenome layers: PENDING fresh run (committed/old: DNASE +1.336, CEBPA +2.777, H3K27ac +1.267, CAGE +1.522).
- BPNet CEBPB head: PENDING (draft +1.99).

### BPNet CEBPB HepG2 (fresh, BP000275): raw=+2.52 (+strand)/+2.39(-) (ref~3 → alt~21)
   draft "BPNet TF head +1.99" → direction ✅ strong gain; magnitude +2.5 vs +1.99 off ~0.5 (no committed BPNet ref existed). Draft didn't specify CEBP member/strand; this is CEBPB.
### AlphaGenome track count: metadata loads 5731 ✅ (backgrounds cover 5168 subset).
### ChromBPNet DNASE cell types incl IMR-90 (lung fibroblast!) → Step 2 ChromBPNet IS testable via IMR-90.
### LegNet panel = K562/HepG2/WTC11 — no lung fibroblast ✅; draft says "HepG2 or A549" but A549 NOT in LegNet ⚠️.
### rs9504151 = chr6:4577675 T/A (GRCh38). No LDLINK_TOKEN → LD/credible-set claims COULD-NOT-TEST.

## MAJOR: ChromBPNet windowing artifact + cell-type screen
- analyze_variant_multilayer uses _auto_region = 1bp region (pos:pos+1). For ChromBPNet this yields ref=48.5, log2FC=+0.318.
- Explicit variant-centered 2114bp window yields ref=287.9 → alt=747.7, **log2FC=+1.374** ≈ draft/committed +1.24.
- ⇒ Draft ChromBPNet +1.24 REPRODUCES (with proper centering). The MCP _auto_region path is a likely REGRESSION/BUG (gives +0.318) to flag to devs.
- Cell-type screen (variant-centered, raw log2FC): HepG2 +1.374 (q.9995), K562 +0.271 (q.97), IMR-90 +2.021 (q.9994), GM12878/H1 pending.
  �eEPInformer ⚠️ "largest in HepG2" questionable: IMR-90 (lung fibroblast) raw +2.02 > HepG2 +1.37. By quantile ~tied. Needs softening.

## Step 1 AlphaGenome FRESH (1Mb JAX/CPU) — reproduces committed closely:
- DNASE:HepG2 +1.333 (draft 1.34 ✅), CEBPA +2.767 (draft 2.7 ✅), CEBPB +3.046, H3K27ac +1.264 (draft 1.27 ✅), CAGE/- +1.525 (draft 1.47 → off 0.055), SORT1 RNA polyA max lnFC +0.591 (~1.8x, <2-fold, up ✅).
- No-prior TF scan top: CEBPB 3.05, CEBPA 2.77, CEBPG 2.27, HLF 2.19, CEBPD 1.82, NFIL3 1.42, ATF4 1.42 → C/EBP family dominates ✅✅.
- Cell-type screen final: HepG2 +1.374, K562 +0.271, IMR-90 +2.021, H1 +0.063, (GM12878 errored/NA). LARGEST=IMR-90 not HepG2 (raw); flag "largest in HepG2" needs softening.

## Step 2 fresh:
- ChromBPNet IMR-90 (lung fibroblast) DNASE at rs9504151 (chr6:4577675 T>A): log2FC=**-0.985** (ref 4655.6→alt 2351.2, q0.993). Active CRE (high ref) + accessibility DROPS on alt ✅. (HepG2 -0.978, K562 -0.355.)
- AG rs9504151: scanning 1886 tracks (lung-fib layers + all 1664 TF for ATF4) — running.
- LD/credible set + cross-oracle ranking: COULD-NOT-TEST (no LDLINK_TOKEN).

## Step 2 AlphaGenome (chr6:4577675 T>A, 1839 tracks, JAX/CPU):
- Lung fibroblast CL:0002553: DNASE -1.35, H3K27ac -1.20 (other fibroblast lines CL:00025xx: DNASE -2.0..-2.4). Accessibility + active histone DROP on alt ✅ (active CRE).
- CDYL/RNA gene_expression: max |lnFC|=0.009 (~1.009x) = NO significant change ✅.
- No-prior TF scan (binding LOST on alt, all negative): CEBPB -3.99, ATF4 -3.97, CEBPB -3.49, CEBPA -3.12, ATF3 -3.00... ATF4 co-top (CEBPB edges it by 0.02). Draft "ATF4 highest, CEBPB high" → reproduces (minor order nuance) ✅.
- ChromBPNet IMR-90 (lung fib): DNASE -0.985 drop on alt ✅.

## Step 2 LD / ranking:
- Credible set (chorus LDlink, r2>0.8): CEU 54 unique (snvs), EUR 49. NOT 18. Many at r2=1.0; no clean rank-2 at 0.9. ⇒ draft "18 variants / rank-2 r2≈0.9" does NOT reproduce.
- FINDING T2: fine_map auto-LD proxies get genome-MISMATCHED ref alleles (many "Provided reference allele X does not match genome" warnings) ⇒ auto-LD fine_map ranking is UNRELIABLE; needs manual ld_variants. rs9504151 itself OK.
- Full cross-oracle ranking (rs9504151 #1; rank-2→19/4): COULD-NOT-RELIABLY-TEST (allele mismatch undermines proxy scoring; AG full-set also slow).
- ISM convergence on ATF4 motif: not independently run (AG TF-track evidence already implicates ATF4/CEBP). 

## MCP fix: added memory guard in alphagenome.py _predict (host-RAM-aware cap, CHORUS_AG_MAX_PREDICT_GB). Blocks all-5731@1Mb (48GB>41GB) -> clear ValueError instead of OOM-killing server. Allows 545/1839-track runs.
## Mac window: 1MB works via JAX/CPU (used for SORT1); ≤600kb = MPS sweet spot for alphagenome_pt (PT). Not a hard cap.

## FIXES APPLIED (2026-06-17) for reproducible Analysis B / LDlink:
- T2 (LD alleles): chorus/utils/ld.py _extract_allele_pairs now uses each proxy's OWN alleles (Alleles column) not the (sentinel,proxy) Correlated_Alleles mis-pairing; chorus/analysis/causal.py prioritize_causal_variants orients ref->genome base before scoring. VERIFIED: re-run fine_map shows 0 genome-mismatch warnings (was ~50). LD proxies now: rs6913171 G/A, rs9502199 T/C, rs4960004 C/T...
- T1 (fine_map window): causal.py region changed from 1bp (pos:pos+1) to oracle.sequence_length window centered on variant (ChromBPNet 2114 -> correct strong effects; AG 1MB). NOT changed: MCP _auto_region (still 1bp; Step1 ChromBPNet via analyze_variant_multilayer still collapses — deferred, needs per-oracle care to avoid Enformer output-window mismatch).
- Re-running rs9504151 fine_map: ChromBPNet IMR-90 (lung-fib, fast) + AlphaGenome (focused lung-fib tracks, ~1.5hr) to get correct ranking. PENDING.
