#!/usr/bin/env python3
"""
Example demonstrating gene expression analysis using CAGE signal at TSS.

This example shows how to:
1. Predict CAGE signal for a genomic region
2. Analyze predicted gene expression by examining CAGE signal at TSS
3. Compare expression between wild-type and variant sequences
"""

import chorus
from chorus.utils import get_genome, download_gencode, get_gene_tss
from chorus.utils.visualization import visualize_chorus_predictions
import numpy as np
import matplotlib.pyplot as plt


def analyze_gata1_expression():
    """Analyze GATA1 expression using CAGE predictions."""
    
    print("=" * 80)
    print("Gene Expression Analysis with Chorus")
    print("=" * 80)
    
    # Setup
    print("\n1. Setting up resources...")
    genome_path = get_genome('hg38')
    gtf_path = download_gencode(version='v48', annotation_type='basic')
    
    # Create oracle
    print("\n2. Loading Enformer model...")
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Define GATA1 region
    gene_name = 'GATA1'
    region = "chrX:48777634-48790694"
    chrom, coords = region.split(':')
    start, end = map(int, coords.split('-'))
    
    print(f"\n3. Analyzing {gene_name} region: {region}")
    
    # Get TSS information
    tss_info = get_gene_tss(gene_name)
    print(f"\nFound {len(tss_info)} transcripts for {gene_name}:")
    for _, tss in tss_info.head(3).iterrows():
        print(f"  - {tss['transcript_id']} TSS at {tss['chrom']}:{tss['tss']} ({tss['strand']})")
    
    # Make predictions - focus on CAGE tracks
    print("\n4. Predicting CAGE signal...")
    cage_tracks = ['CNhs11250', 'CNhs12336']  # K562 CAGE tracks
    wt_predictions = oracle.predict((chrom, start, end), cage_tracks)
    
    # Analyze gene expression
    print("\n5. Analyzing predicted gene expression...")
    expression_analysis = oracle.analyze_gene_expression(
        predictions=wt_predictions,
        gene_name=gene_name,
        chrom=chrom,
        start=start,
        end=end,
        gtf_file=str(gtf_path),
        cage_track_ids=cage_tracks
    )
    
    print(f"\nExpression analysis for {gene_name}:")
    print(f"  Number of TSS in region: {expression_analysis['n_tss']}")
    print(f"  TSS positions: {expression_analysis['tss_positions']}")
    
    print("\n  Predicted CAGE expression:")
    for track_id, mean_expr in expression_analysis['mean_expression'].items():
        max_expr = expression_analysis['max_expression'][track_id]
        print(f"    {track_id}: mean={mean_expr:.2f}, max={max_expr:.2f}")
    
    # Now test a variant effect
    print("\n6. Testing variant effect on expression...")
    
    # Create a variant that might affect expression
    # Let's test a strong enhancer insertion instead of deletion
    variant_region = "chrX:48783000-48783200"  # 200bp region
    print(f"  Testing enhancer replacement at: {variant_region}")
    
    # Create a strong synthetic enhancer sequence
    enhancer_seq = "GATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAGATAAG"[:200]
    
    # Predict with enhancer replacement
    deletion_results = oracle.predict_region_replacement(
        genomic_region=variant_region,
        seq=enhancer_seq,  # Strong GATA motif repeats
        assay_ids=cage_tracks,
        create_tracks=False,
        genome=str(genome_path)
    )
    
    # Analyze expression with deletion
    var_start, var_end = 48783000, 48783200
    deletion_expression = oracle.analyze_gene_expression(
        predictions=deletion_results['raw_predictions'],
        gene_name=gene_name,
        chrom=chrom,
        start=(var_start + var_end) // 2,  # Use variant center
        end=(var_start + var_end) // 2,
        gtf_file=str(gtf_path),
        cage_track_ids=cage_tracks
    )
    
    # Compare expression
    print("\n7. Expression comparison (WT vs Enhancer):")
    for track_id in cage_tracks:
        if track_id in expression_analysis['mean_expression']:
            wt_expr = expression_analysis['mean_expression'][track_id]
            enh_expr = deletion_expression['mean_expression'].get(track_id, 0)
            change = ((enh_expr - wt_expr) / wt_expr * 100) if wt_expr > 0 else 0
            print(f"  {track_id}:")
            print(f"    WT expression: {wt_expr:.2f}")
            print(f"    Enhancer expression: {enh_expr:.2f}")
            print(f"    Change: {change:+.1f}%")
    
    # Visualize the results
    print("\n8. Creating visualization...")
    
    # Get output window coordinates
    region_center = (start + end) // 2
    output_start, output_end = oracle.get_output_window_coords(region_center)
    
    # Create visualization with genes
    visualize_chorus_predictions(
        predictions=wt_predictions,
        chrom=chrom,
        start=output_start,
        track_ids=cage_tracks,
        output_file='gata1_cage_expression.png',
        bin_size=128,
        gtf_file=str(gtf_path),
        use_pygenometracks=True
    )
    print("  Saved visualization to gata1_cage_expression.png")
    
    # Create a simple bar plot of expression changes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    track_names = []
    wt_values = []
    del_values = []
    
    for track_id in cage_tracks:
        if track_id in expression_analysis['mean_expression']:
            track_names.append(track_id)
            wt_values.append(expression_analysis['mean_expression'][track_id])
            del_values.append(deletion_expression['mean_expression'].get(track_id, 0))
    
    x = np.arange(len(track_names))
    width = 0.35
    
    ax.bar(x - width/2, wt_values, width, label='Wild-type', color='blue')
    ax.bar(x + width/2, del_values, width, label='GATA Enhancer', color='red')
    
    ax.set_xlabel('CAGE Track')
    ax.set_ylabel('Mean Expression at TSS')
    ax.set_title(f'{gene_name} Expression Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(track_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gata1_expression_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved expression comparison to gata1_expression_comparison.png")
    
    print("\nDone!")


def test_multiple_genes():
    """Test expression analysis for multiple genes in a region."""
    
    print("\n" + "=" * 80)
    print("Multi-gene Expression Analysis")
    print("=" * 80)
    
    # This would analyze multiple genes in the same region
    # Useful for understanding regulatory effects on gene neighborhoods
    
    # Example genes near GATA1
    genes_to_test = ['GATA1', 'HDAC6', 'GAGE12F']
    
    print("\nAnalyzing expression for multiple genes:")
    for gene in genes_to_test:
        print(f"  - {gene}")
    
    # Implementation would follow similar pattern as above
    # but analyze each gene separately


if __name__ == "__main__":
    # Run the main analysis
    analyze_gata1_expression()
    
    # Optionally test multiple genes
    # test_multiple_genes()