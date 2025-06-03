#!/usr/bin/env python3
"""
Comprehensive Chorus Example: GATA1 Regulatory Analysis with Enformer

This script demonstrates all major features of the Chorus library using the 
Enformer oracle to analyze the GATA1 transcription start site (TSS) region.

GATA1 is an essential transcription factor for red blood cell development.

For an interactive version with visualizations, see gata1_comprehensive_analysis.ipynb
"""

import chorus
from chorus.utils import get_genome, extract_sequence, download_gencode
import numpy as np
import pandas as pd
from pathlib import Path

print("Chorus Comprehensive Example: GATA1 Analysis")
print("=" * 60)

# 1. Setup and initialization
print("\n1. Setting up Enformer oracle...")
genome_path = get_genome('hg38')
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))
oracle.load_pretrained_model()

# Download gene annotations
print("\n   Downloading gene annotations...")
gtf_path = download_gencode(version='v48', annotation_type='basic')

# Define tracks to analyze
# Note: ENCODE track IDs are specific to Enformer/Borzoi
track_ids = ['ENCFF413AHU', 'CNhs11250']  # DNase:K562, CAGE:K562
print(f"   Tracks: {track_ids}")

# 2. Wild-type sequence prediction
print("\n2. Analyzing wild-type GATA1 TSS region...")
gata1_region = "chrX:48777634-48790694"
wt_seq = extract_sequence(gata1_region, str(genome_path))
print(f"   Region: {gata1_region} ({len(wt_seq):,} bp)")
print(f"   GC content: {(wt_seq.count('G') + wt_seq.count('C')) / len(wt_seq) * 100:.1f}%")

# Make predictions
wt_results = oracle.predict(
    ('chrX', 48777634, 48790694),
    track_ids
)

# Print statistics
for track_id, predictions in wt_results.items():
    print(f"   {track_id}: mean={np.mean(predictions):.4f}, max={np.max(predictions):.4f}")

# Save wild-type tracks
print("\n   Saving wild-type predictions...")
wt_files = oracle.save_predictions_as_bedgraph(
    predictions=wt_results,
    chrom='chrX',
    start=48777634,
    end=48790694,
    output_dir="outputs",
    prefix='gata1_wt'
)

# 3. Gene expression analysis
print("\n3. Analyzing GATA1 gene expression...")
expression_analysis = oracle.analyze_gene_expression(
    predictions=wt_results,
    gene_name='GATA1',
    chrom='chrX',
    start=48777634,
    end=48790694,
    gtf_file=str(gtf_path),
    cage_track_ids=['CNhs11250']
)

print(f"   TSS positions found: {expression_analysis['n_tss']}")
print(f"   CAGE expression at TSS:")
for track_id, expr in expression_analysis['mean_expression'].items():
    print(f"   - {track_id}: mean={expr:.2f}, max={expression_analysis['max_expression'][track_id]:.2f}")

# 4. Region replacement
print("\n4. Testing region replacement with synthetic enhancer...")
replace_region = "chrX:48782929-48783129"
replacement_seq = "GATA" * 50  # 200bp of GATA motifs
print(f"   Replacing {replace_region} with GATA repeat enhancer")

replacement_results = oracle.predict_region_replacement(
    genomic_region=replace_region,
    seq=replacement_seq,
    assay_ids=track_ids,
    genome=str(genome_path)
)

# Compare with wild-type
print("   Effect of replacement:")
for track_id in track_ids:
    wt_mean = np.mean(wt_results[track_id])
    new_mean = np.mean(replacement_results['raw_predictions'][track_id])
    print(f"   - {track_id}: {wt_mean:.4f} → {new_mean:.4f} (Δ={new_mean-wt_mean:+.4f})")

# 5. Sequence insertion
print("\n5. Testing sequence insertion...")
insertion_pos = "chrX:48781929"
insert_seq = replacement_seq  # Same GATA enhancer
print(f"   Inserting GATA enhancer at {insertion_pos}")

insertion_results = oracle.predict_region_insertion_at(
    genomic_position=insertion_pos,
    seq=insert_seq,
    assay_ids=track_ids,
    genome=str(genome_path)
)

# Analyze impact
print("   Impact of insertion:")
for track_id in track_ids:
    wt_mean = np.mean(wt_results[track_id])
    new_mean = np.mean(insertion_results['raw_predictions'][track_id])
    print(f"   - {track_id}: {wt_mean:.4f} → {new_mean:.4f} (Δ={new_mean-wt_mean:+.4f})")

# 6. Variant effect analysis
print("\n6. Testing variant effects...")
variant_pos = 48786129
ref_seq = 'C'
alt_alleles = ['A', 'G', 'T']
print(f"   Testing SNP at chrX:{variant_pos} (ref={ref_seq})")

variant_results = oracle.predict_variant_effect(
    genomic_region=f"chrX:{variant_pos-5000}-{variant_pos+5000}",
    variant_position=f"chrX:{variant_pos}",
    alleles=[ref_seq] + alt_alleles,
    assay_ids=track_ids,
    genome=str(genome_path)
)

# Analyze effects
print("   Variant effects:")
for track_id in track_ids:
    ref_signal = np.mean(variant_results['predictions']['reference'][track_id])
    print(f"\n   {track_id}:")
    print(f"   - Reference ({ref_seq}): {ref_signal:.4f}")
    
    for i, alt in enumerate(alt_alleles):
        alt_key = f'alt_{i+1}'
        alt_signal = np.mean(variant_results['predictions'][alt_key][track_id])
        effect = alt_signal - ref_signal
        print(f"   - {ref_seq}→{alt}: {alt_signal:.4f} (Δ={effect:+.4f})")

# 7. Synthetic sequence prediction
print("\n7. Testing fully synthetic sequence...")
# Create synthetic sequence with known elements
context_size = 393216  # Enformer requirement
synthetic_seq = ('TATAAA' + 'N' * 20 + 'CCAAT' + 'N' * 50 + 'GGGCGG' + 'N' * 100) * 100
synthetic_seq = (synthetic_seq + 'ACGT' * 1000)[:context_size]
print(f"   Created synthetic sequence with promoter elements ({len(synthetic_seq):,} bp)")

synthetic_results = oracle.predict(synthetic_seq, track_ids)

# Analyze predictions
print("   Synthetic sequence predictions:")
for track_id, predictions in synthetic_results.items():
    print(f"   - {track_id}: mean={np.mean(predictions):.4f}, max={np.max(predictions):.4f}")

# Summary
print("\n" + "=" * 60)
print("Analysis complete!")
print(f"\nOutput files saved to: outputs/")
print("\nKey findings:")
print("- GATA1 shows strong CAGE signal at TSS positions")
print("- GATA motif insertions can modulate regulatory activity")
print("- Single nucleotide changes can have measurable effects")
print("\nFor interactive visualizations, run:")
print("jupyter notebook gata1_comprehensive_analysis.ipynb")