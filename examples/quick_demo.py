#!/usr/bin/env python3
"""
Quick demonstration of Chorus key features.

This script shows all five main prediction methods in a concise format.
For detailed analysis with visualizations, see gata1_comprehensive_analysis.ipynb
"""

import chorus
from chorus.utils import get_genome
import numpy as np

# Setup
print("Setting up Chorus with Enformer...")
genome_path = get_genome('hg38')
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))
oracle.load_pretrained_model()

# Define tracks
tracks = ['ENCFF413AHU', 'CNhs11250']  # DNase:K562, CAGE:K562
print(f"Predicting tracks: {tracks}\n")

# 1. Wild-type prediction
print("1. Wild-type prediction")
wt_predictions = oracle.predict(
    ('chr11', 5247000, 5248000),  # Beta-globin locus
    tracks
)
for track, values in wt_predictions.items():
    print(f"   {track}: mean={np.mean(values):.2f}, max={np.max(values):.2f}")

# 2. Region replacement
print("\n2. Region replacement (200bp enhancer)")
enhancer = 'GATA' * 50  # 200bp of GATA motifs
replaced = oracle.predict_region_replacement(
    'chr11:5247400-5247600',
    enhancer,
    tracks
    # genome parameter not needed - uses oracle's reference_fasta
)
for track in tracks:
    wt_mean = np.mean(wt_predictions[track])
    new_mean = np.mean(replaced['raw_predictions'][track])
    print(f"   {track}: {wt_mean:.2f} → {new_mean:.2f} (Δ={new_mean-wt_mean:+.2f})")

# 3. Sequence insertion
print("\n3. Sequence insertion at chr11:5247500")
inserted = oracle.predict_region_insertion_at(
    'chr11:5247500',
    enhancer,
    tracks
    # genome parameter not needed - uses oracle's reference_fasta
)
for track in tracks:
    wt_mean = np.mean(wt_predictions[track])
    new_mean = np.mean(inserted['raw_predictions'][track])
    print(f"   {track}: {wt_mean:.2f} → {new_mean:.2f} (Δ={new_mean-wt_mean:+.2f})")

# 4. Variant effect
print("\n4. Variant effect analysis (SNP at chr11:5247500)")
# Note: The reference allele at chr11:5247500 is 'C'
variant = oracle.predict_variant_effect(
    'chr11:5247000-5248000',
    'chr11:5247500',
    ['C', 'A', 'G', 'T'],  # Reference 'C' first, then alternates
    tracks
    # genome parameter not needed - uses oracle's reference_fasta
)
print("   Effect sizes vs reference 'C':")
for track in tracks:
    effects = variant['effect_sizes']
    print(f"   {track}:")
    for i, alt in enumerate(['alt_1', 'alt_2', 'alt_3']):
        allele = ['A', 'G', 'T'][i]
        effect = np.mean(effects[alt][track])
        print(f"      C→{allele}: {effect:+.4f}")

# 5. Save predictions
print("\n5. Saving predictions as BedGraph...")
oracle.save_predictions_as_bedgraph(
    wt_predictions,
    chrom='chr11',
    start=5247000,
    end=5248000,  # Add end coordinate to avoid warning
    output_dir='outputs',
    prefix='betaglobin_demo'
)
print("   Saved to outputs/")

print("\nDone! For detailed analysis with visualizations, run:")
print("jupyter notebook gata1_comprehensive_analysis.ipynb")