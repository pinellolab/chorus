#!/usr/bin/env python3
"""
Quick test of GATA1 examples to verify they work correctly.
"""

import chorus
from chorus.utils import get_genome

# Test basic setup
print("Testing Chorus GATA1 examples...")

# Get genome
genome_path = get_genome('hg38')
print(f"✓ Genome path: {genome_path}")

# Create oracle
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))
oracle.load_pretrained_model()
print("✓ Oracle loaded")

# Test each example briefly
print("\nTesting Example A: Wild-type prediction...")
wt_results = oracle.predict(
    ('chrX', 48777634, 48790694),
    ['DNase:K562']
)
print(f"  Result shape: {wt_results['DNase:K562'].shape}")

print("\nTesting Example B: Region replacement...")
results = oracle.predict_region_replacement(
    genomic_region="chrX:48782929-48783129",
    seq="ACGT" * 50,  # 200bp replacement
    assay_ids=['DNase:K562'],
    genome=str(genome_path)
)
print(f"  Result keys: {list(results.keys())}")

print("\nTesting Example C: Sequence insertion...")
results = oracle.predict_region_insertion_at(
    genomic_position="chrX:48781929",
    seq="GAATTC",  # 6bp insertion
    assay_ids=['DNase:K562'],
    genome=str(genome_path)
)
print(f"  Result keys: {list(results.keys())}")

print("\nTesting Example D: Variant analysis...")
results = oracle.predict_variant_effect(
    genomic_region="chrX:48778229-48788229",
    variant_position="chrX:48783229",
    alleles=['T', 'A'],  # T is reference, A is alternate
    assay_ids=['DNase:K562'],
    genome=str(genome_path)
)
print(f"  Predictions available: {list(results['predictions'].keys())}")

print("\nTesting Example E: Direct sequence...")
seq = "ACGT" * 98304  # 393,216 bp
results = oracle.predict(seq, ['DNase:K562'])
print(f"  Result shape: {results['DNase:K562'].shape}")

print("\n✅ All examples work correctly!")