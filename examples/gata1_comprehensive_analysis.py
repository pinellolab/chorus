#!/usr/bin/env python3
"""
Comprehensive example of Chorus library usage with Enformer oracle.

This example demonstrates all major prediction capabilities using the GATA1 
transcription start site (TSS) region on chromosome X as a test case.

GATA1 is an essential transcription factor for red blood cell development,
making it an interesting target for regulatory analysis.
"""

import chorus
from chorus.utils import get_genome, extract_sequence, download_gencode
from chorus.utils.visualization import visualize_chorus_predictions
import numpy as np
from pathlib import Path


# ==============================================================================
# INSTALLATION INSTRUCTIONS (for reference - assuming already installed)
# ==============================================================================
"""
To install Chorus, follow these steps:

1. Clone the repository:
   git clone https://github.com/pinellolab/chorus.git
   cd chorus

2. Create the main Chorus environment:
   mamba env create -f environment.yml
   mamba activate chorus

3. Install Chorus package:
   pip install -e .

4. Set up the Enformer environment:
   chorus setup --oracle enformer

5. Download the reference genome:
   chorus genome download hg38

6. (Optional) For advanced track visualization:
   pip install pyGenomeTracks
"""


def setup_oracle_and_genome():
    """Set up the Enformer oracle and download genome if needed."""
    print("=" * 80)
    print("CHORUS - Comprehensive Example with GATA1 TSS Region")
    print("=" * 80)
    
    # Step 1: Get reference genome (auto-downloads if not present)
    print("\n1. Setting up reference genome...")
    genome_path = get_genome('hg38')
    print(f"   Using genome: {genome_path}")
    
    # Step 2: Download gene annotations
    print("\n2. Setting up gene annotations...")
    gtf_path = download_gencode(version='v48', annotation_type='basic')
    print(f"   Using annotations: {gtf_path}")
    
    # Step 3: Create and configure Enformer oracle
    print("\n3. Creating Enformer oracle with environment isolation...")
    oracle = chorus.create_oracle(
        'enformer', 
        use_environment=True,
        reference_fasta=str(genome_path)
    )
    
    # Step 4: Load pre-trained model
    print("\n4. Loading pre-trained Enformer model...")
    oracle.load_pretrained_model()
    print("   Model loaded successfully!")
    
    return oracle, genome_path, gtf_path


def print_available_assays(oracle):
    """Print available assay types and cell types."""
    print("\n5. Available tracks in Enformer:")
    print("-" * 40)
    
    # Get assay types
    assay_types = oracle.list_assay_types()
    print(f"\nAvailable assay types ({len(assay_types)}):")
    for i, assay in enumerate(assay_types, 1):
        print(f"   {i:2d}. {assay}")
    
    # Get cell types
    cell_types = oracle.list_cell_types()
    print(f"\nAvailable cell types ({len(cell_types)}):")
    # Show just first 10 due to large number
    for i, cell in enumerate(cell_types[:10], 1):
        print(f"   {i:2d}. {cell}")
    print(f"   ... and {len(cell_types) - 10} more")
    
    # Get track summary
    track_summary = oracle.get_track_info()
    print("\nTrack summary by assay type:")
    for assay, count in track_summary.items():
        print(f"   {assay}: {count} tracks")
    
    # Show specific K562 tracks
    print("\nK562 tracks available:")
    dnase_k562 = oracle.get_track_info("DNASE:K562")
    cage_k562 = oracle.get_track_info("CAGE:.*K562")
    print(f"   DNASE:K562 - {len(dnase_k562)} tracks")
    if len(dnase_k562) > 0:
        print(f"     Example: {dnase_k562.iloc[0]['identifier']} - {dnase_k562.iloc[0]['description']}")
    print(f"   CAGE with K562 - {len(cage_k562)} tracks")
    if len(cage_k562) > 0:
        print(f"     Example: {cage_k562.iloc[0]['identifier']} - {cage_k562.iloc[0]['description']}")


def example_a_wildtype_prediction(oracle, genome_path, gtf_path):
    """Example A: Predict wild-type sequence values."""
    print("\n" + "=" * 80)
    print("Example A: Wild-type Sequence Prediction")
    print("=" * 80)
    
    # Define GATA1 TSS region
    region = "chrX:48777634-48790694"
    print(f"\nAnalyzing wild-type region: {region}")
    print("This region contains the GATA1 transcription start site")
    
    # Extract sequence info
    seq = extract_sequence(region, str(genome_path))
    print(f"Region length: {len(seq):,} bp")
    
    # Make predictions using ENCODE identifiers for better specificity
    print("\nMaking predictions using specific track identifiers...")
    # Use ENCFF413AHU for DNase:K562 and CNhs11250 for CAGE:K562
    track_ids = ['ENCFF413AHU', 'CNhs11250']
    print(f"  Track IDs: {track_ids}")
    
    results = oracle.predict(
        ('chrX', 48777634, 48790694),
        track_ids
    )
    
    # Print statistics
    for track_id, predictions in results.items():
        print(f"\n{track_id}:")
        print(f"  Shape: {predictions.shape}")
        print(f"  Mean signal: {np.mean(predictions):.4f}")
        print(f"  Max signal: {np.max(predictions):.4f}")
        print(f"  Signal at TSS region (bins 40-50): {np.mean(predictions[40:50]):.4f}")
    
    # Save tracks using oracle method - Enformer handles coordinate mapping internally
    print("\nSaving tracks to disk...")
    saved_files = oracle.save_predictions_as_bedgraph(
        predictions=results,
        chrom='chrX',
        start=48777634,
        end=48790694,  # Provide end coordinate for proper mapping
        output_dir="bedgraph_outputs",
        prefix='a_wt'
    )
    print(f"  Saved {len(saved_files)} files")
    
    # For visualization, get the output window coordinates
    region_center = (48777634 + 48790694) // 2
    output_start, output_end = oracle.get_output_window_coords(region_center)
    print(f"\nEnformer output window: chrX:{output_start}-{output_end}")
    
    # Visualize with improved plotting and gene annotations
    print("\nCreating visualization with gene annotations...")
    visualize_chorus_predictions(
        predictions=results,
        chrom='chrX',
        start=output_start,  # Use output window start for visualization
        track_ids=track_ids,
        output_file='a_wt_visualization_with_genes.png',
        bin_size=128,
        gtf_file=str(gtf_path),
        use_pygenometracks=True
    )
    print("  Visualization saved to a_wt_visualization_with_genes.png")
    
    return results


def example_b_region_replacement(oracle, genome_path, gtf_path):
    """Example B: Replace a sub-region with synthetic sequence."""
    print("\n" + "=" * 80)
    print("Example B: Region Replacement")
    print("=" * 80)
    
    # Define replacement
    replace_region = "chrX:48782929-48783129"
    replacement_seq = "CTGCTTGCTTTAGCTTCAGGGTTCTTATCTTTTTTCATTTTATAACAGCAAAGGCGACACCCAACATGTGCGTGCTTGAGATAATGACTAAAAACTGCCCGTGACTCAAGCGCTTCTGGTGAGGGAAGATAAGGCAAGGAAACTGGCCGCCTAGATAGCCCTGGGAATGAGGCAGTCTCTGTTCTGGGTAAAGTGTCTGC"
    
    print(f"\nReplacing region: {replace_region}")
    print(f"Replacement length: {len(replacement_seq)} bp")
    print(f"Original region is {200} bp, replacement is {len(replacement_seq)} bp")
    
    # Make prediction with replacement
    print("\nMaking predictions with replaced region...")
    results = oracle.predict_region_replacement(
        genomic_region=replace_region,
        seq=replacement_seq,
        assay_ids=['ENCFF413AHU', 'CNhs11250'],  # Using specific track IDs
        create_tracks=False,  # We'll save manually for custom naming
        genome=str(genome_path)
    )
    
    # Analyze changes
    print("\nAnalyzing signal changes due to replacement:")
    for track_id in ['ENCFF413AHU', 'CNhs11250']:
        predictions = results['normalized_scores'][track_id]
        print(f"\n{track_id}:")
        print(f"  Mean signal: {np.mean(predictions):.4f}")
        print(f"  Max signal: {np.max(predictions):.4f}")
    
    # Save tracks
    print("\nSaving replacement tracks...")
    replace_start, replace_end = 48782929, 48783129
    
    saved_files = oracle.save_predictions_as_bedgraph(
        predictions=results['raw_predictions'],
        chrom='chrX',
        start=replace_start,
        end=replace_end,  # Provide end coordinate
        output_dir="bedgraph_outputs",
        prefix='b_replacement'
    )
    print(f"  Saved {len(saved_files)} files")
    
    # Get output window for visualization
    replace_center = (replace_start + replace_end) // 2
    output_start, _ = oracle.get_output_window_coords(replace_center)
    
    # Visualize replacement results
    print("\nCreating replacement visualization...")
    visualize_chorus_predictions(
        predictions=results['raw_predictions'],
        chrom='chrX',
        start=output_start,  # Use output window start
        track_ids=['ENCFF413AHU', 'CNhs11250'],
        output_file='b_replacement_visualization.png',
        bin_size=128
    )
    
    return results


def example_c_sequence_insertion(oracle, genome_path, gtf_path):
    """Example C: Insert sequence at specific position."""
    print("\n" + "=" * 80)
    print("Example C: Sequence Insertion")
    print("=" * 80)
    
    # Define insertion
    insertion_pos = "chrX:48781929"
    insert_seq = "CTGCTTGCTTTAGCTTCAGGGTTCTTATCTTTTTTCATTTTATAACAGCAAAGGCGACACCCAACATGTGCGTGCTTGAGATAATGACTAAAAACTGCCCGTGACTCAAGCGCTTCTGGTGAGGGAAGATAAGGCAAGGAAACTGGCCGCCTAGATAGCCCTGGGAATGAGGCAGTCTCTGTTCTGGGTAAAGTGTCTGC"
    
    print(f"\nInserting at position: {insertion_pos}")
    print(f"Insert sequence length: {len(insert_seq)} bp")
    print("This insertion is ~1kb upstream of the main promoter")
    
    # Make prediction with insertion
    print("\nMaking predictions with insertion...")
    results = oracle.predict_region_insertion_at(
        genomic_position=insertion_pos,
        seq=insert_seq,
        assay_ids=['ENCFF413AHU', 'CNhs11250'],  # Using specific track IDs
        create_tracks=False,
        genome=str(genome_path)
    )
    
    # Analyze impact
    print("\nAnalyzing impact of insertion:")
    for track_id in ['ENCFF413AHU', 'CNhs11250']:
        predictions = results['normalized_scores'][track_id]
        # Find peak around insertion site (center of output)
        center = len(predictions) // 2
        window = 20
        local_signal = predictions[center-window:center+window]
        
        print(f"\n{track_id}:")
        print(f"  Overall mean: {np.mean(predictions):.4f}")
        print(f"  Signal around insertion: {np.mean(local_signal):.4f}")
        print(f"  Peak near insertion: {np.max(local_signal):.4f}")
    
    # Save tracks
    print("\nSaving insertion tracks...")
    insert_position = 48781929
    
    saved_files = oracle.save_predictions_as_bedgraph(
        predictions=results['raw_predictions'],
        chrom='chrX',
        start=insert_position,
        end=insert_position,  # For insertions, start=end at insertion point
        output_dir="bedgraph_outputs",
        prefix='c_insertion'
    )
    print(f"  Saved {len(saved_files)} files")
    
    # Get output window for visualization
    output_start, _ = oracle.get_output_window_coords(insert_position)
    
    # Visualize insertion results
    print("\nCreating insertion visualization...")
    visualize_chorus_predictions(
        predictions=results['raw_predictions'],
        chrom='chrX',
        start=output_start,  # Use output window start
        track_ids=['ENCFF413AHU', 'CNhs11250'],
        output_file='c_insertion_visualization.png',
        bin_size=128
    )
    
    return results


def example_d_variant_analysis(oracle, genome_path, gtf_path):
    """Example D: Analyze all possible SNPs at a position."""
    print("\n" + "=" * 80)
    print("Example D: Variant Effect Analysis")
    print("=" * 80)
    
    # Define variant position
    variant_pos = 48783229
    
    print(f"\nAnalyzing all possible variants at position: chrX:{variant_pos}")
    
    # Get reference allele
    ref_seq = extract_sequence(f"chrX:{variant_pos}-{variant_pos+1}", str(genome_path))
    print(f"Reference allele: {ref_seq}")
    
    # Test all possible substitutions
    # Create alleles list with reference first, then alternatives
    alt_alleles = [a for a in ['A', 'C', 'G', 'T'] if a != ref_seq]
    test_alleles = [ref_seq] + alt_alleles  # Reference first, then alternatives
    
    print(f"\nTesting substitutions: {ref_seq} -> {', '.join(alt_alleles)}")
    
    # Predict effects for all alleles
    print("\nPredicting variant effects...")
    results = oracle.predict_variant_effect(
        genomic_region=f"chrX:{variant_pos-5000}-{variant_pos+5000}",  # Need wider region
        variant_position=f"chrX:{variant_pos}",
        alleles=test_alleles,  # Reference first, then alternatives
        assay_ids=['ENCFF413AHU', 'CNhs11250'],  # Using specific track IDs
        genome=str(genome_path)
    )
    
    # Analyze effects
    print("\nVariant effect analysis:")
    print("-" * 60)
    
    # The reference is always at index 0 in test_alleles
    ref_idx = 0
    
    track_ids = ['ENCFF413AHU', 'CNhs11250']
    for i, track_id in enumerate(track_ids):
        print(f"\n{track_id}:")
        
        # Get reference signal
        ref_key = 'reference'
        ref_signal = np.mean(results['predictions'][ref_key][:, i])
        print(f"  Reference ({ref_seq}) mean signal: {ref_signal:.4f}")
        
        # Compare all variants
        print("  Variant effects:")
        for j, allele in enumerate(alt_alleles):
            allele_key = f'alt_{j+1}'
            if allele_key in results['predictions']:
                alt_signal = np.mean(results['predictions'][allele_key][:, i])
                effect = alt_signal - ref_signal
                print(f"    {allele}: {alt_signal:.4f} (Δ = {effect:+.4f})")
    
    # Save tracks for each variant
    print("\nSaving variant tracks...")
    
    # Save reference
    ref_preds = {
        'ENCFF413AHU': results['predictions']['reference'][:, 0],
        'CNhs11250': results['predictions']['reference'][:, 1]
    }
    oracle.save_predictions_as_bedgraph(
        predictions=ref_preds,
        chrom='chrX',
        start=variant_pos,
        end=variant_pos,  # Single position variant
        output_dir="bedgraph_outputs",
        prefix=f'd_variant_{ref_seq}'
    )
    
    # Save alternatives
    for i, allele in enumerate(alt_alleles):
        allele_key = f'alt_{i+1}'
        if allele_key in results['predictions']:
            # Create dict with track names
            variant_preds = {
                'ENCFF413AHU': results['predictions'][allele_key][:, 0],
                'CNhs11250': results['predictions'][allele_key][:, 1]
            }
            oracle.save_predictions_as_bedgraph(
                predictions=variant_preds,
                chrom='chrX',
                start=variant_pos,
                end=variant_pos,  # Single position variant
                output_dir="bedgraph_outputs",
                prefix=f'd_variant_{allele}'
            )
    
    # Get output window for visualization
    output_start, _ = oracle.get_output_window_coords(variant_pos)
    
    # Visualize variant effects
    print("\nCreating variant visualization...")
    # Create combined visualization showing reference and one variant
    combined_preds = {
        f'ENCFF413AHU_ref_{ref_seq}': results['predictions']['reference'][:, 0],
        f'ENCFF413AHU_alt_{alt_alleles[0]}': results['predictions']['alt_1'][:, 0],
        f'CNhs11250_ref_{ref_seq}': results['predictions']['reference'][:, 1],
        f'CNhs11250_alt_{alt_alleles[0]}': results['predictions']['alt_1'][:, 1]
    }
    visualize_chorus_predictions(
        predictions=combined_preds,
        chrom='chrX',
        start=output_start,  # Use output window start
        track_ids=list(combined_preds.keys()),
        output_file='d_variant_comparison_visualization.png',
        bin_size=128
    )
    
    return results


def example_e_direct_sequence_prediction(oracle, gtf_path):
    """Example E: Direct sequence prediction without genomic coordinates."""
    print("\n" + "=" * 80)
    print("Example E: Direct Sequence Prediction")
    print("=" * 80)
    
    print("\nCreating a synthetic sequence to demonstrate direct prediction...")
    
    # Create a synthetic sequence with known regulatory elements
    # Must be exactly 393,216 bp for Enformer
    context_size = 393216
    
    # Build sequence with regulatory elements
    promoter_elements = {
        'TATA_box': 'TATAAA',
        'CAAT_box': 'CCAAT',
        'GC_box': 'GGGCGG',
        'GATA_motif': 'GATA',
        'E_box': 'CACGTG'
    }
    
    # Create base sequence
    print("Building synthetic sequence with regulatory elements:")
    seq_parts = []
    
    # Add repeated regulatory elements every 1kb
    for i in range(0, context_size, 1000):
        if i % 5000 == 0:
            # Add strong promoter
            motifs = ''.join([
                promoter_elements['TATA_box'],
                'N' * 20,
                promoter_elements['CAAT_box'],
                'N' * 50,
                promoter_elements['GC_box']
            ])
            seq_parts.append(motifs)
            print(f"  Position {i}: Added promoter elements")
        elif i % 3000 == 0:
            # Add enhancer elements
            enhancer = (promoter_elements['GATA_motif'] + 'NNNN') * 5
            seq_parts.append(enhancer)
        else:
            # Random sequence
            seq_parts.append('ACGT' * 250)
    
    # Combine and trim to exact size
    full_seq = ''.join(seq_parts)[:context_size]
    
    # Pad if needed
    if len(full_seq) < context_size:
        full_seq += 'A' * (context_size - len(full_seq))
    
    print(f"\nFinal sequence length: {len(full_seq):,} bp")
    
    # Make prediction
    print("\nMaking predictions on synthetic sequence...")
    results = oracle.predict(
        full_seq,
        ['ENCFF413AHU', 'CNhs11250']  # Using specific track IDs
    )
    
    # Analyze predictions
    print("\nAnalyzing synthetic sequence predictions:")
    for track_id, predictions in results.items():
        print(f"\n{track_id}:")
        print(f"  Shape: {predictions.shape}")
        print(f"  Mean signal: {np.mean(predictions):.4f}")
        print(f"  Max signal: {np.max(predictions):.4f}")
        
        # Find peaks (top 5 bins)
        top_bins = np.argsort(predictions)[-5:]
        print(f"  Top 5 peak positions (bins): {top_bins}")
    
    # Save tracks
    print("\nSaving synthetic sequence tracks...")
    # Use arbitrary coordinates for visualization
    saved_files = oracle.save_predictions_as_bedgraph(
        predictions=results,
        chrom='synthetic',
        start=0,
        output_dir="bedgraph_outputs",
        prefix='e_synthetic'
    )
    print(f"  Saved {len(saved_files)} files")
    
    # Visualize synthetic results
    print("\nCreating synthetic sequence visualization...")
    visualize_chorus_predictions(
        predictions=results,
        chrom='synthetic',
        start=0,
        track_ids=['ENCFF413AHU', 'CNhs11250'],
        output_file='e_synthetic_visualization.png',
        bin_size=128
    )
    
    return results




def main():
    """Run all examples."""
    # Check environment exists
    from chorus.core.environment import EnvironmentManager
    env_manager = EnvironmentManager()
    
    if not env_manager.environment_exists('enformer'):
        print("\nERROR: Enformer environment not set up.")
        print("Please run: chorus setup --oracle enformer")
        return
    
    try:
        # Setup
        oracle, genome_path, gtf_path = setup_oracle_and_genome()
        print_available_assays(oracle)
        
        # Run all examples
        wt_results = example_a_wildtype_prediction(oracle, genome_path, gtf_path)
        replacement_results = example_b_region_replacement(oracle, genome_path, gtf_path)
        insertion_results = example_c_sequence_insertion(oracle, genome_path, gtf_path)
        variant_results = example_d_variant_analysis(oracle, genome_path, gtf_path)
        synthetic_results = example_e_direct_sequence_prediction(oracle, gtf_path)
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nGenerated BedGraph files:")
        print("  a_wt_*.bedgraph - Wild-type predictions")
        print("  b_replacement_*.bedgraph - Region replacement predictions")
        print("  c_insertion_*.bedgraph - Sequence insertion predictions")
        print("  d_variant_*.bedgraph - Variant effect predictions")
        print("  e_synthetic_*.bedgraph - Direct sequence predictions")
        print("\nThese files can be loaded into genome browsers for visualization.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()