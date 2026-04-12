"""Validate per-bin normalization by plotting raw vs normalized signal.

Generates matplotlib PNGs showing raw and percentile-normalized signal
for representative tracks at a known locus.  Used to visually verify
that peaks stand out and background noise maps to low percentiles.

Run in chorus env:
  mamba run -n chorus python scripts/validate_perbin_normalization.py
"""
import sys
import os
sys.path.insert(0, '/PHShome/lp698/chorus')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chorus.analysis.normalization import get_pertrack_normalizer


def validate_perbin(oracle_name='enformer'):
    """Load normalizer and plot CDF diagnostics for key tracks."""
    norm = get_pertrack_normalizer(oracle_name)
    if norm is None:
        print(f"No per-track normalizer found for '{oracle_name}'")
        return

    entry = norm._loaded[oracle_name]
    perbin_cdfs = entry['perbin_cdfs']
    summary_cdfs = entry['summary_cdfs']

    from chorus.oracles.enformer_source.enformer_metadata import EnformerMetadata
    meta = EnformerMetadata()
    df = meta.tracks_df

    # Pick representative tracks
    tracks_to_check = {
        'CAGE': df[df['description'].str.startswith('CAGE:')].iloc[0],
        'DNASE': df[df['description'].str.startswith('DNASE:')].iloc[0],
        'H3K27ac': df[df['description'].str.contains('H3K27ac', na=False)].iloc[0],
        'CTCF': df[df['description'].str.contains('CTCF', na=False)].iloc[0],
        'H3K4me3': df[df['description'].str.contains('H3K4me3', na=False)].iloc[0],
    }

    out_dir = '/PHShome/lp698/chorus/logs'

    # ── Plot 1: Perbin CDF shapes ──
    fig, axes = plt.subplots(len(tracks_to_check), 1, figsize=(12, 3 * len(tracks_to_check)))
    for ax, (label, row) in zip(axes, tracks_to_check.items()):
        tid = row['identifier']
        idx = entry['track_index'].get(str(tid))
        if idx is None:
            ax.set_title(f'{label}: NOT FOUND')
            continue
        cdf = perbin_cdfs[idx]
        quantiles = np.linspace(0, 1, len(cdf))

        ax.plot(quantiles, cdf, linewidth=1.5)
        ax.set_title(f'{label} ({tid}): {row["description"][:60]}')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Raw bin value')
        ax.set_xlim(0, 1)

        # Mark key percentiles
        for p in [0.5, 0.9, 0.95, 0.99]:
            val = cdf[int(p * len(cdf))]
            ax.axhline(val, color='red', alpha=0.3, linestyle='--')
            ax.text(0.02, val, f'p{int(p*100)}={val:.2f}', fontsize=8, color='red')

    plt.tight_layout()
    cdf_path = os.path.join(out_dir, 'perbin_cdf_shapes.png')
    plt.savefig(cdf_path, dpi=150)
    plt.close()
    print(f'Saved: {cdf_path}')

    # ── Plot 2: What percentile does near-zero map to? ──
    fig, ax = plt.subplots(figsize=(10, 5))
    test_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    for label, row in tracks_to_check.items():
        tid = str(row['identifier'])
        idx = entry['track_index'].get(tid)
        if idx is None:
            continue
        cdf = perbin_cdfs[idx]
        pctiles = [np.searchsorted(cdf, v, side='right') / len(cdf) for v in test_values]
        ax.plot(range(len(test_values)), pctiles, 'o-', label=label, markersize=6)

    ax.set_xticks(range(len(test_values)))
    ax.set_xticklabels([str(v) for v in test_values])
    ax.set_xlabel('Raw bin value')
    ax.set_ylabel('Percentile')
    ax.set_title('Raw value → Percentile mapping per track type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    mapping_path = os.path.join(out_dir, 'perbin_value_mapping.png')
    plt.savefig(mapping_path, dpi=150)
    plt.close()
    print(f'Saved: {mapping_path}')

    # ── Print summary stats ──
    print()
    print('=== PERBIN CDF SUMMARY ===')
    for label, row in tracks_to_check.items():
        tid = str(row['identifier'])
        idx = entry['track_index'].get(tid)
        if idx is None:
            continue
        cdf = perbin_cdfs[idx]
        count = entry.get('perbin_counts')
        n = count[idx] if count is not None else len(cdf)
        print(f'{label:12s} ({tid}): p50={cdf[5000]:.4f}, p90={cdf[9000]:.4f}, '
              f'p95={cdf[9500]:.4f}, p99={cdf[9900]:.4f}, max={cdf[-1]:.2f}, '
              f'n_samples={n}')

    # Key check: near-zero should map to LOW percentile
    print()
    print('=== NEAR-ZERO MAPPING (should be <0.30 for all) ===')
    for label, row in tracks_to_check.items():
        tid = str(row['identifier'])
        idx = entry['track_index'].get(tid)
        if idx is None:
            continue
        cdf = perbin_cdfs[idx]
        rank = np.searchsorted(cdf, 0.03, side='right')
        pctile = rank / len(cdf)
        status = '✓' if pctile < 0.30 else '✗ TOO HIGH'
        print(f'  {label:12s}: raw=0.03 → pctile={pctile:.4f} {status}')


if __name__ == '__main__':
    validate_perbin()
