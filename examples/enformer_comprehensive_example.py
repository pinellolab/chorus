#!/usr/bin/env python3
"""
Comprehensive example demonstrating Enformer oracle usage with Chorus.

This example shows:
1. Basic sequence prediction
2. Genomic coordinate prediction with reference genome
3. Using ENCODE identifiers vs descriptive track names
4. Creating BedGraph track files for visualization
"""

import chorus
import numpy as np
import sys
import os

def example_basic_sequence_prediction():
    """Example 1: Basic sequence prediction without reference genome."""
    print("\n" + "="*80)
    print("Example 1: Basic Sequence Prediction")
    print("="*80)
    
    # Create oracle without reference genome
    oracle = chorus.create_oracle('enformer', use_environment=True)
    oracle.load_pretrained_model()
    
    # Create a random sequence (normally you'd use a real sequence)
    sequence = 'ACGT' * 1000  # 4000 bp sequence
    
    # Predict using descriptive track name
    predictions = oracle.predict(sequence, ['DNase:K562'])
    
    print(f"Input sequence length: {len(sequence)} bp")
    print(f"Predictions shape: {predictions['DNase:K562'].shape}")
    print(f"Mean predicted signal: {predictions['DNase:K562'].mean():.4f}")
    

def example_genomic_coordinate_prediction(reference_fasta):
    """Example 2: Predict using genomic coordinates with reference padding."""
    print("\n" + "="*80)
    print("Example 2: Genomic Coordinate Prediction with Reference Genome")
    print("="*80)
    
    # Define region of interest
    region = "chrX:48780505-48785229"
    chrom, coords = region.split(':')
    start, end = map(int, coords.split('-'))
    
    # Create oracle with reference genome
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True, 
                                 reference_fasta=reference_fasta)
    oracle.load_pretrained_model()
    
    # Predict using ENCODE identifier
    track_id = "ENCFF413AHU"  # DNase:K562 specific experiment
    predictions = oracle.predict((chrom, start, end), [track_id])
    
    print(f"Region: {region}")
    print(f"Track: {track_id} (DNase:K562)")
    print(f"Predictions shape: {predictions[track_id].shape}")
    print(f"Mean predicted signal: {predictions[track_id].mean():.4f}")
    
    # Create BedGraph file
    create_bedgraph_track(predictions[track_id], chrom, start, end, 
                         track_id, oracle.sequence_length)
    

def example_multiple_tracks(reference_fasta):
    """Example 3: Predict multiple tracks simultaneously."""
    print("\n" + "="*80)
    print("Example 3: Multiple Track Prediction")
    print("="*80)
    
    region = "chr1:1000000-1100000"
    chrom, coords = region.split(':')
    start, end = map(int, coords.split('-'))
    
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=reference_fasta)
    oracle.load_pretrained_model()
    
    # Request multiple tracks
    tracks = ['DNase:K562', 'DNase:HepG2', 'DNase:GM12878']
    predictions = oracle.predict((chrom, start, end), tracks)
    
    print(f"Region: {region}")
    print(f"Requested {len(tracks)} tracks")
    
    for track in tracks:
        print(f"\n{track}:")
        print(f"  Shape: {predictions[track].shape}")
        print(f"  Mean: {predictions[track].mean():.4f}")
        print(f"  Max: {predictions[track].max():.4f}")


def create_bedgraph_track(predictions, chrom, start, end, track_name, 
                         input_length=393216):
    """Helper function to create BedGraph file from predictions."""
    
    # Calculate coordinate mapping
    output_length = 896 * 128  # 114,688 bp
    output_offset = (input_length - output_length) // 2
    region_center = (start + end) // 2
    input_center = input_length // 2
    
    # Map region to output coordinates
    region_start_in_input = input_center - (region_center - start)
    region_end_in_input = input_center + (end - region_center)
    region_start_in_output = region_start_in_input - output_offset
    region_end_in_output = region_end_in_input - output_offset
    
    # Convert to bins
    start_bin = max(0, region_start_in_output // 128)
    end_bin = min(896, (region_end_in_output + 127) // 128)
    
    # Create filename
    bedgraph_file = f"{track_name}_{chrom}_{start}_{end}.bedgraph"
    
    with open(bedgraph_file, 'w') as f:
        # Write header
        f.write(f'track type=bedGraph name="{track_name}" ')
        f.write(f'description="Enformer prediction for {track_name}" ')
        f.write(f'visibility=full autoScale=on color=255,0,0\n')
        
        # Write data
        for i in range(start_bin, end_bin):
            bin_start_in_output = i * 128
            bin_pos_in_region = bin_start_in_output - region_start_in_output
            bin_start_genomic = start + bin_pos_in_region
            bin_end_genomic = bin_start_genomic + 128
            
            # Clip to region bounds
            bin_start_genomic = max(start, bin_start_genomic)
            bin_end_genomic = min(end, bin_end_genomic)
            
            if bin_start_genomic < bin_end_genomic:
                value = float(predictions[i])
                f.write(f"{chrom}\t{bin_start_genomic}\t{bin_end_genomic}\t")
                f.write(f"{value:.6f}\n")
    
    print(f"\nCreated BedGraph file: {os.path.abspath(bedgraph_file)}")
    return bedgraph_file


def main():
    """Run all examples."""
    print("\nChorus Enformer Oracle - Comprehensive Examples")
    print("=" * 80)
    
    # Check if reference FASTA is provided
    reference_fasta = None
    if len(sys.argv) > 1:
        reference_fasta = sys.argv[1]
        if not os.path.exists(reference_fasta):
            print(f"Error: Reference FASTA not found at {reference_fasta}")
            sys.exit(1)
        print(f"Using reference genome: {reference_fasta}")
    else:
        print("No reference genome provided.")
        print("Usage: python enformer_comprehensive_example.py [/path/to/hg38.fa]")
        print("\nRunning only Example 1 (sequence-based prediction)...")
    
    # Example 1: Always run basic sequence prediction
    example_basic_sequence_prediction()
    
    # Examples 2 & 3: Only run if reference genome is provided
    if reference_fasta:
        example_genomic_coordinate_prediction(reference_fasta)
        example_multiple_tracks(reference_fasta)
    else:
        print("\nSkipping Examples 2 & 3 (require reference genome)")
        print("Provide path to hg38.fa to run all examples")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()