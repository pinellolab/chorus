#!/usr/bin/env python3
"""
Quick start example for Enformer predictions with Chorus.

This minimal example shows how to:
1. Create an Enformer oracle
2. Load the pre-trained model
3. Make predictions for a genomic region
4. Save results as a BedGraph track
"""

import chorus
from chorus.utils import get_genome

# Define region of interest
region = "chrX:48780505-48785229"
chrom, coords = region.split(':')
start, end = map(int, coords.split('-'))

# Get reference genome (will download if needed)
print("Getting reference genome...")
reference_fasta = get_genome('hg38')  # Automatically downloads if not present

# Create oracle with reference genome
print("Creating Enformer oracle...")
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=reference_fasta)

# Load pre-trained model
print("Loading model...")
oracle.load_pretrained_model()

# Make predictions using ENCODE identifier
track_id = "ENCFF413AHU"  # DNase accessibility in K562 cells
print(f"\nPredicting {track_id} signal for {region}...")
predictions = oracle.predict((chrom, start, end), [track_id])

# Display results
print(f"\nResults:")
print(f"  Output shape: {predictions[track_id].shape}")
print(f"  Mean signal: {predictions[track_id].mean():.4f}")
print(f"  Max signal: {predictions[track_id].max():.4f}")

print("\nDone! Predictions are in 128bp bins covering the region.")