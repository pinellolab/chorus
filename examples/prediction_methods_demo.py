#!/usr/bin/env python3
"""
Demo of the three main prediction methods in Chorus.

This script demonstrates:
1. Sequence replacement - Replace a genomic region with new sequence
2. Sequence insertion - Insert sequence at a specific position  
3. Variant effect - Predict effects of genetic variants
"""

import numpy as np
import chorus
from chorus.utils import get_genome
import matplotlib.pyplot as plt


def demo_sequence_replacement():
    """Demo: Replace a promoter region with a synthetic sequence."""
    print("\n=== SEQUENCE REPLACEMENT DEMO ===")
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', use_environment=True)
    
    # Get reference genome
    genome_path = get_genome('hg38')
    oracle.reference_fasta = str(genome_path)
    
    # Load model
    print("Loading Enformer model...")
    oracle.load_pretrained_model()
    
    # Define a promoter region to replace (MYC promoter)
    region = "chr8:127735000-127740000"  # 5kb region
    
    # Create a synthetic promoter sequence
    # Enformer requires exactly 393,216 bp, so we'll pad around our 5kb insert
    context_size = 393216
    synthetic_promoter = "ACGT" * 1250  # 5kb synthetic sequence
    
    # For this demo, we'll use the predict method with sequence directly
    # First, extract the full context sequence
    from chorus.utils.sequence import extract_sequence
    
    # Extract centered context around the region
    center = (127735000 + 127740000) // 2
    start = center - context_size // 2
    end = center + context_size // 2
    
    # Get the original sequence
    original_seq = extract_sequence(f"chr8:{start}-{end}", str(genome_path))
    
    # Replace the middle 5kb with synthetic sequence
    replace_start = context_size // 2 - 2500
    replace_end = context_size // 2 + 2500
    modified_seq = (original_seq[:replace_start] + 
                   synthetic_promoter + 
                   original_seq[replace_end:])
    
    # Make predictions
    print(f"Predicting for original sequence...")
    original_pred = oracle.predict(original_seq, ['DNase:K562'])
    
    print(f"Predicting for modified sequence...")
    modified_pred = oracle.predict(modified_seq, ['DNase:K562'])
    
    # Compare predictions
    diff = modified_pred['DNase:K562'] - original_pred['DNase:K562']
    
    print(f"\nPrediction shapes: {original_pred['DNase:K562'].shape}")
    print(f"Max effect size: {np.max(np.abs(diff)):.3f}")
    print(f"Effect region: bins {np.argmax(np.abs(diff))-10} to {np.argmax(np.abs(diff))+10}")
    
    return original_pred, modified_pred, diff


def demo_sequence_insertion():
    """Demo: Insert an enhancer at a specific position."""
    print("\n=== SEQUENCE INSERTION DEMO ===")
    
    # Create oracle with reference genome
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    
    # Load model
    print("Loading Enformer model...")
    oracle.load_pretrained_model()
    
    # Insert a 1kb enhancer sequence upstream of a gene
    enhancer_seq = "GGATCC" * 167  # ~1kb with BamHI sites
    insertion_pos = "chr8:127730000"  # 5kb upstream of MYC
    
    # Use the insertion method
    print(f"Inserting {len(enhancer_seq)}bp enhancer at {insertion_pos}")
    results = oracle.predict_region_insertion_at(
        genomic_position=insertion_pos,
        seq=enhancer_seq,
        assay_ids=['DNase:K562', 'H3K27ac:K562'],
        genome=str(genome_path)
    )
    
    print(f"\nPrediction results:")
    print(f"Raw predictions shape: {results['raw_predictions'].shape}")
    print(f"Number of tracks: {len(results['track_objects'])}")
    
    # Analyze the effect around insertion site
    # The insertion is at the center of the output window
    center_bin = results['raw_predictions'].shape[0] // 2
    window = 20  # Look at +/- 20 bins around insertion
    
    for i, track in enumerate(results['track_objects']):
        signal = results['raw_predictions'][center_bin-window:center_bin+window, i]
        print(f"{track.name} - Mean signal around insertion: {np.mean(signal):.3f}")
    
    return results


def demo_variant_effect():
    """Demo: Predict effects of genetic variants."""
    print("\n=== VARIANT EFFECT DEMO ===")
    
    # Create oracle
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    
    # Load model
    print("Loading Enformer model...")
    oracle.load_pretrained_model()
    
    # Define a variant in a regulatory region
    # This would typically come from a VCF file
    variant_region = "chr8:127735434-127735434"  # MYC promoter variant
    variant_pos = "chr8:127735434"
    ref_allele = "G"
    alt_alleles = ["A", "C", "T"]  # Multi-allelic variant
    
    print(f"Analyzing variant at {variant_pos}: {ref_allele} > {','.join(alt_alleles)}")
    
    # Predict variant effects
    results = oracle.predict_variant_effect(
        genomic_region=variant_region,
        variant_position=variant_pos,
        alleles=[ref_allele] + alt_alleles,
        assay_ids=['DNase:K562', 'CTCF:K562'],
        genome=str(genome_path)
    )
    
    print(f"\nVariant effect results:")
    print(f"Predictions available for: {list(results['predictions'].keys())}")
    
    # Analyze effects
    ref_pred = results['predictions']['allele_0']  # Reference allele
    
    for i, alt in enumerate(alt_alleles, 1):
        alt_pred = results['predictions'][f'allele_{i}']
        effect = results['effect_sizes'][f'allele_{i}']
        
        # Find the bin with maximum effect
        max_effect_bin = np.argmax(np.abs(effect[:, 0]))  # First track
        max_effect = effect[max_effect_bin, 0]
        
        print(f"\n{ref_allele}>{alt} variant:")
        print(f"  Max effect size: {max_effect:.3f} at bin {max_effect_bin}")
        print(f"  Mean absolute effect: {np.mean(np.abs(effect)):.3f}")
    
    return results


def demo_batch_variant_analysis():
    """Demo: Analyze multiple variants from a VCF-like format."""
    print("\n=== BATCH VARIANT ANALYSIS DEMO ===")
    
    # Create oracle
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    
    # Load model
    print("Loading Enformer model...")
    oracle.load_pretrained_model()
    
    # Define multiple variants (would typically be loaded from VCF)
    variants = [
        {
            'chrom': 'chr8',
            'pos': 127735434,
            'ref': 'G',
            'alt': ['A'],
            'id': 'rs1234567'
        },
        {
            'chrom': 'chr8', 
            'pos': 127736000,
            'ref': 'CT',
            'alt': ['C'],  # Deletion
            'id': 'rs2345678'
        },
        {
            'chrom': 'chr8',
            'pos': 127737000, 
            'ref': 'A',
            'alt': ['ATTT'],  # Insertion
            'id': 'rs3456789'
        }
    ]
    
    print(f"Analyzing {len(variants)} variants...")
    
    # Analyze each variant
    all_effects = []
    
    for var in variants:
        # Convert to format expected by predict_variant_effect
        region = f"{var['chrom']}:{var['pos']}-{var['pos']}"
        position = f"{var['chrom']}:{var['pos']}"
        alleles = [var['ref']] + var['alt']
        
        print(f"\nAnalyzing {var['id']}: {var['ref']}>{','.join(var['alt'])}")
        
        try:
            results = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=position,
                alleles=alleles,
                assay_ids=['DNase:K562'],
                genome=str(genome_path)
            )
            
            # Extract effect size
            effect = results['effect_sizes']['allele_1']  # First alt allele
            max_effect = np.max(np.abs(effect))
            
            all_effects.append({
                'variant_id': var['id'],
                'max_effect': max_effect,
                'mean_effect': np.mean(np.abs(effect))
            })
            
            print(f"  Max effect: {max_effect:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Rank variants by effect size
    if all_effects:
        print("\n=== VARIANT RANKING BY EFFECT SIZE ===")
        sorted_effects = sorted(all_effects, key=lambda x: x['max_effect'], reverse=True)
        for i, var in enumerate(sorted_effects, 1):
            print(f"{i}. {var['variant_id']}: max={var['max_effect']:.3f}, mean={var['mean_effect']:.3f}")
    
    return all_effects


def main():
    """Run all demos."""
    print("Chorus Prediction Methods Demo")
    print("==============================")
    
    # Check if enformer environment exists
    from chorus.core.environment import EnvironmentManager
    env_manager = EnvironmentManager()
    
    if not env_manager.environment_exists('enformer'):
        print("\nERROR: Enformer environment not set up.")
        print("Please run: chorus setup --oracle enformer")
        return
    
    # Run demos
    try:
        # 1. Sequence replacement
        demo_sequence_replacement()
        
        # 2. Sequence insertion
        demo_sequence_insertion()
        
        # 3. Variant effect (single)
        demo_variant_effect()
        
        # 4. Batch variant analysis
        demo_batch_variant_analysis()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()