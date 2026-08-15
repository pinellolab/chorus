# Checklist §3, §4, §12, §13 — the four sections the new-user audit did not cover, 2026-08-15

Run against `9ad4cce` (= `v0.7.3`). GPUs 0–3 idle at start; other tenants held 12–16 GiB on 4–7.

## §3 GPU / device detection — PASS

Base env: `linux_x86_64_cuda has_cuda=True`.

All nine oracle envs see a device:

| env | probe |
|---|---|
| `chorus-borzoi` | torch 2.12.1, `cuda=True` |
| `chorus-cherimoya` | torch 2.13.0+cu130, `cuda=True` |
| `chorus-epinformerseq` | torch 2.13.0+cu130, `cuda=True` |
| `chorus-sei` | torch 2.12.1, `cuda=True` |
| `chorus-legnet` | torch 2.5.1.post303, `cuda=True` |
| `chorus-alphagenome_pt` | torch 2.13.0+cu130, `cuda=True` |
| `chorus-alphagenome` | jax 0.10.2, `dev=gpu` |
| `chorus-enformer` | tf 2.13.1 — **0 GPUs bare, 1 GPU via the runner path** |
| `chorus-chrombpnet` | tf 2.8.0 — **0 GPUs bare, 1 GPU via the runner path** |

The two TF envs are the checklist's own documented trap, and it caught me: their CUDA libs live in
`nvidia-*-cu11` pip wheels that only `EnvironmentRunner._prepare_env` puts on `LD_LIBRARY_PATH`
(9 entries, first `…/site-packages/nvidia/curand/lib`). Exporting that path turns `GPUS 0` into
`GPUS 1` for both. A bare probe cannot distinguish "no GPU support" from "GPU only on the runner
path" — which is exactly what §3 says, and why it says it.

## §4 Per-track CDF / normalization — PASS, every published figure exact

Ran the checklist's own script plus the track-count and signed-fraction tables:

| oracle | tracks | published | signed | published | mono | p50≤p95≤p99 | perbin |
|---|---|---|---|---|---|---|---|
| alphagenome | 5,168 | 5,168 | 0.129 | ~0.129 | ✓ | ✓ | yes |
| borzoi | 7,611 | 7,611 | 0.203 | ~0.203 | ✓ | ✓ | yes |
| cherimoya | 1,518 | 1,518 | 0.000 | 0 | ✓ | ✓ | yes |
| chrombpnet | 753 | 753 | 0.000 | 0 | ✓ | ✓ | yes |
| enformer | 5,313 | 5,313 | 0.000 | 0 | ✓ | ✓ | yes |
| epinformerseq | 33 | 33 | 0.000 | 0 | ✓ | ✓ | no (by design) |
| legnet | 3 | 3 | 1.000 | 1.0 | ✓ | ✓ | no (by design) |
| sei | 40 | 40 | 1.000 | 1.0 | ✓ | ✓ | no (by design) |

Eight of eight load without `None`; no effect-CDF row is non-monotonic; every `perbin_cdfs`
present/absent matches the documented split.

## §13 Scientific determinism — same-process PASS, cross-process reproduces F8

**Same input twice in one process: 4 of 4 bitwise identical**, one from each framework family.

| oracle | framework | track | shape | result |
|---|---|---|---|---|
| enformer | TF | `ENCFF413AHU` | (896,) | `bitwise=True`, `max diff 0.000e+00` |
| chrombpnet | TF | `DNASE:K562` | (2114,) | `bitwise=True`, `max diff 0.000e+00` |
| sei | PyTorch | `TA#HeLa_…Cervix` | (1,) | `bitwise=True`, `max diff 0.000e+00` |
| legnet | PyTorch | `LentiMPRA:HepG2` | (20,) | `bitwise=True`, `max diff 0.000e+00` |

**End-to-end cross-process: `scripts/gate_end_to_end_determinism.py --oracle enformer --strings`
FAILS**, worst relative delta **3.441e-02**, with a `quantile_score` moving 0.6005 → 0.6069 and
`raw_score`/`alt_value` differing across `all_scores`.

That is not a new finding: it is **F8**, written up in
[`audits/2026-08-14_f8_localisation.md`](2026-08-14_f8_localisation.md) with a measured `raw_score`
max of 4.29%, and it remains open and unlocalised. The value of running both halves here is that they
reproduce F8's defining asymmetry in one sitting — *bitwise identical in a quiet in-process probe,
3.4% apart between two real cross-process runs*. Anyone who checks determinism the way I did first
(two `predict()` calls, one process) will conclude the oracle is deterministic and be wrong about the
path `regenerate_examples.py` actually uses.

## §12 Reproducibility — partially exercised

- **Notebook codegen is idempotent**: verified over three consecutive runs, 0 dirty files
  ([#218](https://github.com/pinellolab/chorus/pull/218)).
- **The `*_variant_report.pkl` trap is real and correctly gitignored** (`.gitignore:148`). All four
  are present on this host, so `--consolidate` would not degrade here; a fresh clone has none, which
  is the condition the checklist warns about. `tests/test_rerender_refuses_to_degrade.py` is the guard
  that makes that degradation crash rather than ship.
- **Not run:** full regeneration of the committed walkthroughs. That is hours of GPU across per-oracle
  envs, and F8 above means the Enformer example is known not to reproduce bit-for-bit anyway. Stated
  rather than quietly skipped.

## A finding that is not in the checklist: track listing has four different shapes

Getting a track id for a determinism probe took four attempts, because each oracle exposes it
differently:

| oracle | how you get a track |
|---|---|
| enformer | pass a known id (`ENCFF413AHU`) |
| sei | `oracle._get_all_assay_ids()` |
| legnet | `predict(region)` with `assay_ids=None` |
| chrombpnet | `load_pretrained_model(assay="DNASE", cell_type="K562")` |

`get_track_info()` does not exist on any of them; `_get_all_assay_ids()` exists on sei but not legnet;
`assay=` is a `load_pretrained_model` argument, not a constructor one. Every one of those is
discoverable only by reading that oracle's source. For a new user this is the difference between
"list what I can predict" being one call and being a per-oracle research task, and for a contributor
adding a ninth oracle there is no single method to implement. Not fixed here — it is an API change,
not a doc fix — but worth a decision rather than another audit noting it.

## Coverage after this pass

14 sections were covered by the new-user audit earlier today; these four bring it to 18 of 18. The
only checklist items deliberately left unexercised are the full walkthrough regeneration (§12, hours,
and F8 defeats byte-comparison) and cross-machine drift (§13, needs a second machine).
