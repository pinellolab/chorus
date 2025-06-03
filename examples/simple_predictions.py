#!/usr/bin/env python3
"""
Simple examples of the three main prediction methods in Chorus.
This script shows the basic usage without complex analysis.
"""

import chorus
from chorus.utils import get_genome


def example_1_sequence_prediction():
    """Example 1: Basic sequence prediction."""
    print("\n=== Example 1: Sequence Prediction ===")
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', use_environment=True)
    oracle.load_pretrained_model()
    
    # Create a test sequence (must be exactly 393,216 bp for Enformer)
    sequence = "ACGT" * 98304  # 393,216 bp
    
    # Predict
    predictions = oracle.predict(sequence, ['DNase:K562'])
    
    print(f"Input sequence length: {len(sequence)} bp")
    print(f"Prediction shape: {predictions['DNase:K562'].shape}")
    print(f"Number of bins: {len(predictions['DNase:K562'])}")
    print(f"Each bin represents: 128 bp")
    

def example_2_genomic_region_prediction():
    """Example 2: Predict from genomic coordinates."""
    print("\n=== Example 2: Genomic Region Prediction ===")
    
    # Get genome
    genome_path = get_genome('hg38')
    
    # Create oracle with reference genome
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Predict for a genomic region (automatically extracts sequence)
    # Using MYC gene locus
    predictions = oracle.predict(
        ('chr8', 127735434, 127736434),  # 1kb region
        ['DNase:K562', 'ChIP-seq_H3K27ac:K562']
    )
    
    print(f"Predicted tracks: {list(predictions.keys())}")
    print(f"Prediction shape: {predictions['DNase:K562'].shape}")
    

def example_3_variant_effect():
    """Example 3: Simple variant effect prediction."""
    print("\n=== Example 3: Variant Effect Prediction ===")
    
    # Get genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Define a simple SNP
    # The API requires the variant to be within a genomic region
    # We'll analyze a 10kb region around the variant
    variant_pos = 127735434
    region_start = variant_pos - 5000
    region_end = variant_pos + 5000
    
    variant_region = f"chr8:{region_start}-{region_end}"
    variant_position = f"chr8:{variant_pos}"
    alleles = ["G", "A"]  # Reference: G, Alternative: A
    
    # Predict variant effect
    results = oracle.predict_variant_effect(
        genomic_region=variant_region,
        variant_position=variant_position,
        alleles=alleles,
        assay_ids=['DNase:K562'],
        genome=str(genome_path)
    )
    
    print(f"Variant: {alleles[0]} > {alleles[1]}")
    print(f"Predictions available for: {list(results['predictions'].keys())}")
    print(f"Effect size shape: {results['effect_sizes']['allele_1'].shape}")
    
    # Get maximum effect
    import numpy as np
    max_effect = np.max(np.abs(results['effect_sizes']['allele_1']))
    print(f"Maximum effect size: {max_effect:.3f}")


def main():
    """Run all examples."""
    print("Simple Chorus Prediction Examples")
    print("=================================")
    
    try:
        example_1_sequence_prediction()
        example_2_genomic_region_prediction()
        example_3_variant_effect()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up the Enformer environment: chorus setup --oracle enformer")
        print("2. Downloaded hg38 genome: chorus genome download hg38")


if __name__ == "__main__":
    main()