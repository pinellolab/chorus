#!/usr/bin/env python3
"""
Examples demonstrating predict_region_replacement and predict_region_insertion_at.

These functions allow you to:
1. Replace an entire genomic region with a new sequence
2. Insert a sequence at a specific genomic position
"""

import numpy as np
import pandas as pd
import chorus
from chorus.utils import get_genome, extract_sequence
import matplotlib.pyplot as plt
from pathlib import Path


def example_region_replacement():
    """Example: Replace a promoter region with a synthetic sequence."""
    print("\n=== REGION REPLACEMENT EXAMPLE ===")
    
    # Get reference genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Define the region to replace (e.g., a promoter)
    # This should be exactly the size that fits Enformer's context
    context_size = 393216  # Enformer's required input size
    center = 127735434  # MYC promoter region center
    
    genomic_region = f"chr8:{center - context_size//2}-{center + context_size//2}"
    
    # Create a synthetic sequence to replace the region
    # For demonstration, we'll create a sequence with:
    # - Strong TATA box motifs
    # - CpG islands
    # - CTCF binding sites
    
    # Build synthetic promoter
    tata_box = "TATAAA"
    gc_rich = "CGCGCG" * 10
    ctcf_motif = "CCGCGNGGNGGCAG"
    
    # Create full replacement sequence
    synthetic_seq = ""
    
    # Add repeated regulatory elements
    for i in range(0, context_size, 1000):
        if i % 5000 == 0:
            synthetic_seq += tata_box + "A" * 20  # TATA box with spacer
        elif i % 3000 == 0:
            synthetic_seq += gc_rich  # GC-rich region
        elif i % 7000 == 0:
            synthetic_seq += ctcf_motif.replace("N", "A")  # CTCF site
        else:
            # Random sequence for the rest
            synthetic_seq += "ACGT" * 250
    
    # Ensure exact length
    synthetic_seq = synthetic_seq[:context_size]
    if len(synthetic_seq) < context_size:
        synthetic_seq += "A" * (context_size - len(synthetic_seq))
    
    print(f"Replacing region: {genomic_region}")
    print(f"Replacement sequence length: {len(synthetic_seq)} bp")
    
    # Perform region replacement
    results = oracle.predict_region_replacement(
        genomic_region=genomic_region,
        seq=synthetic_seq,
        assay_ids=['DNase:K562', 'ChIP-seq_CTCF:K562'],
        create_tracks=True,
        genome=str(genome_path)
    )
    
    print("\nResults:")
    print(f"Raw predictions shape: {results['raw_predictions'].shape}")
    print(f"Normalized scores shape: {results['normalized_scores'].shape}")
    print(f"Number of tracks: {len(results['track_objects'])}")
    
    # Analyze the predictions
    for i, track in enumerate(results['track_objects']):
        scores = results['normalized_scores'][:, i]
        print(f"\n{track.name}:")
        print(f"  Mean signal: {np.mean(scores):.3f}")
        print(f"  Max signal: {np.max(scores):.3f}")
        print(f"  Signal variance: {np.var(scores):.3f}")
    
    # Save track files if created
    if results['track_files']:
        print(f"\nTrack files saved: {results['track_files']}")
    
    return results


def example_region_replacement_with_dataframe():
    """Example: Use DataFrame input for region replacement."""
    print("\n=== REGION REPLACEMENT WITH DATAFRAME ===")
    
    # Get reference genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Create DataFrame with genomic region
    context_size = 393216
    center = 127735434
    
    region_df = pd.DataFrame({
        'chrom': ['chr8'],
        'start': [center - context_size//2],
        'end': [center + context_size//2]
    })
    
    print("Region DataFrame:")
    print(region_df)
    
    # Create enhancer-like sequence
    enhancer_seq = "GGTACC" * 20  # Repeated binding sites
    spacer = "A" * 100
    
    # Build full sequence with enhancer elements
    synthetic_seq = ""
    for i in range(0, context_size, 1000):
        if i % 10000 == 0:
            synthetic_seq += enhancer_seq + spacer
        else:
            synthetic_seq += "ACGT" * 250
    
    synthetic_seq = synthetic_seq[:context_size]
    synthetic_seq += "A" * (context_size - len(synthetic_seq))
    
    # Predict with DataFrame input
    results = oracle.predict_region_replacement(
        genomic_region=region_df,
        seq=synthetic_seq,
        assay_ids=['DNase:K562'],
        create_tracks=False,
        genome=str(genome_path)
    )
    
    print(f"\nPrediction successful!")
    print(f"Output shape: {results['raw_predictions'].shape}")
    
    return results


def example_sequence_insertion():
    """Example: Insert an enhancer sequence at a specific position."""
    print("\n=== SEQUENCE INSERTION EXAMPLE ===")
    
    # Get reference genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Define insertion position (5kb upstream of MYC)
    insertion_position = "chr8:127730434"
    
    # Create a strong enhancer sequence to insert
    # Based on known enhancer motifs
    enhancer_elements = [
        "GGTACC",      # KpnI site (often in enhancers)
        "CAGCTG",      # AP-1 binding site
        "TGACTCA",     # Another AP-1 variant
        "CACGTG",      # MYC binding site (E-box)
        "GGAAA",       # ETS binding site
    ]
    
    # Build 1kb enhancer with multiple binding sites
    enhancer_seq = ""
    for i in range(20):  # Repeat pattern 20 times
        for element in enhancer_elements:
            enhancer_seq += element + "TTTT"  # Add spacer
    
    # Pad to exactly 1kb
    enhancer_seq = enhancer_seq[:1000]
    if len(enhancer_seq) < 1000:
        enhancer_seq += "A" * (1000 - len(enhancer_seq))
    
    print(f"Inserting {len(enhancer_seq)}bp enhancer at {insertion_position}")
    print(f"Enhancer contains {len(enhancer_elements)} different binding motifs")
    
    # Perform insertion
    results = oracle.predict_region_insertion_at(
        genomic_position=insertion_position,
        seq=enhancer_seq,
        assay_ids=['DNase:K562', 'ChIP-seq_H3K27ac:K562'],
        create_tracks=True,
        genome=str(genome_path)
    )
    
    print("\nResults:")
    print(f"Raw predictions shape: {results['raw_predictions'].shape}")
    
    # Analyze impact around insertion site
    # The insertion affects the center of the output window
    center_bin = results['raw_predictions'].shape[0] // 2
    window_size = 50  # Analyze 50 bins around insertion
    
    for i, track in enumerate(results['track_objects']):
        # Get signal around insertion
        start_bin = max(0, center_bin - window_size)
        end_bin = min(results['raw_predictions'].shape[0], center_bin + window_size)
        
        local_signal = results['normalized_scores'][start_bin:end_bin, i]
        
        print(f"\n{track.name} around insertion site:")
        print(f"  Mean signal: {np.mean(local_signal):.3f}")
        print(f"  Peak signal: {np.max(local_signal):.3f}")
        print(f"  Signal range: {np.ptp(local_signal):.3f}")
    
    return results


def example_insertion_with_dataframe():
    """Example: Use DataFrame for insertion position."""
    print("\n=== SEQUENCE INSERTION WITH DATAFRAME ===")
    
    # Get reference genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Create DataFrame with insertion position
    position_df = pd.DataFrame({
        'chrom': ['chr8'],
        'start': [127730434],
        'end': [127730434]  # Same as start for point insertion
    })
    
    print("Position DataFrame:")
    print(position_df)
    
    # Create a simple test sequence
    test_seq = "ATCG" * 250  # 1kb sequence
    
    # Perform insertion
    results = oracle.predict_region_insertion_at(
        genomic_position=position_df,
        seq=test_seq,
        assay_ids=['DNase:K562'],
        create_tracks=False,
        genome=str(genome_path)
    )
    
    print(f"\nInsertion successful!")
    print(f"Output shape: {results['raw_predictions'].shape}")
    
    return results


def compare_original_vs_modified():
    """Compare predictions before and after modifications."""
    print("\n=== COMPARING ORIGINAL VS MODIFIED REGIONS ===")
    
    # Get reference genome
    genome_path = get_genome('hg38')
    
    # Create oracle
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Define region
    context_size = 393216
    center = 127735434
    genomic_region = f"chr8:{center - context_size//2}-{center + context_size//2}"
    
    # Get original sequence and predict
    print("1. Predicting original region...")
    original_seq = extract_sequence(genomic_region, str(genome_path))
    
    original_results = oracle.predict_region_replacement(
        genomic_region=genomic_region,
        seq=original_seq,
        assay_ids=['DNase:K562'],
        genome=str(genome_path)
    )
    
    # Create modified sequence (add strong enhancer in the middle)
    print("2. Creating modified sequence with enhancer...")
    modified_seq = list(original_seq)
    
    # Insert strong enhancer in the middle
    mid_point = len(original_seq) // 2
    enhancer = "CACGTG" * 100  # 600bp of E-box motifs
    
    for i, base in enumerate(enhancer):
        if mid_point + i < len(modified_seq):
            modified_seq[mid_point + i] = base
    
    modified_seq = ''.join(modified_seq)
    
    # Predict modified
    modified_results = oracle.predict_region_replacement(
        genomic_region=genomic_region,
        seq=modified_seq,
        assay_ids=['DNase:K562'],
        genome=str(genome_path)
    )
    
    # Compare results
    print("\n3. Comparing predictions...")
    
    original_signal = original_results['normalized_scores'][:, 0]
    modified_signal = modified_results['normalized_scores'][:, 0]
    
    difference = modified_signal - original_signal
    
    print(f"Original mean signal: {np.mean(original_signal):.3f}")
    print(f"Modified mean signal: {np.mean(modified_signal):.3f}")
    print(f"Maximum change: {np.max(np.abs(difference)):.3f}")
    
    # Find regions with biggest changes
    top_changes = np.argsort(np.abs(difference))[-10:]
    print(f"\nBins with largest changes: {top_changes}")
    
    # Plot if matplotlib available
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(original_signal, label='Original', alpha=0.7)
        plt.plot(modified_signal, label='Modified', alpha=0.7)
        plt.legend()
        plt.ylabel('DNase Signal')
        plt.title('DNase Predictions: Original vs Modified')
        
        plt.subplot(2, 1, 2)
        plt.plot(difference)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Difference (Modified - Original)')
        plt.xlabel('Genomic Bin')
        plt.title('Effect of Enhancer Insertion')
        
        plt.tight_layout()
        plt.savefig('region_modification_comparison.png', dpi=150)
        print(f"\nPlot saved as: region_modification_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    return original_results, modified_results, difference


def main():
    """Run all examples."""
    print("Region Manipulation Examples")
    print("============================")
    
    # Check environment
    from chorus.core.environment import EnvironmentManager
    env_manager = EnvironmentManager()
    
    if not env_manager.environment_exists('enformer'):
        print("\nERROR: Enformer environment not set up.")
        print("Please run: chorus setup --oracle enformer")
        return
    
    # Run examples
    try:
        # 1. Basic region replacement
        example_region_replacement()
        
        # 2. Region replacement with DataFrame
        example_region_replacement_with_dataframe()
        
        # 3. Sequence insertion
        example_sequence_insertion()
        
        # 4. Insertion with DataFrame
        example_insertion_with_dataframe()
        
        # 5. Compare original vs modified
        compare_original_vs_modified()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()