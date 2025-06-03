#!/usr/bin/env python3
"""
Examples showing the flexible region manipulation capabilities.

Both predict_region_replacement and predict_region_insertion_at can handle
sequences of any length - from single bases to large regions.
"""

import chorus
from chorus.utils import get_genome
import numpy as np


def example_snp_simulation():
    """Simulate SNP effects using region replacement."""
    print("\n=== SNP SIMULATION ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Simulate different SNPs at the same position
    position = 127735434
    snps = {
        'A': 'chr8:127735434-127735435',
        'C': 'chr8:127735434-127735435', 
        'G': 'chr8:127735434-127735435',
        'T': 'chr8:127735434-127735435'
    }
    
    results = {}
    for allele, region in snps.items():
        result = oracle.predict_region_replacement(
            genomic_region=region,
            seq=allele,
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        results[allele] = result['normalized_scores']['DNase:K562']
        print(f"Allele {allele}: mean signal = {results[allele].mean():.4f}")
    
    # Find allele with highest activity
    mean_signals = {a: r.mean() for a, r in results.items()}
    best_allele = max(mean_signals, key=mean_signals.get)
    print(f"\nHighest activity allele: {best_allele}")


def example_motif_insertion():
    """Insert various regulatory motifs."""
    print("\n=== MOTIF INSERTION ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Common regulatory motifs
    motifs = {
        'TATA box': 'TATAAA',
        'CAAT box': 'CCAAT',
        'GC box': 'GGGCGG',
        'E-box': 'CACGTG',
        'CTCF': 'CCGCGNGGNGGCAG'.replace('N', 'A'),
        'AP-1': 'TGACTCA',
        'NF-κB': 'GGGACTTTCC'
    }
    
    insertion_pos = 'chr8:127730434'  # 5kb upstream of MYC
    
    for motif_name, motif_seq in motifs.items():
        result = oracle.predict_region_insertion_at(
            genomic_position=insertion_pos,
            seq=motif_seq,
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        
        # Look at signal around insertion (center of output)
        center = len(result['normalized_scores']['DNase:K562']) // 2
        window = 10
        local_signal = result['normalized_scores']['DNase:K562'][center-window:center+window]
        
        print(f"{motif_name} ({motif_seq}): max local signal = {local_signal.max():.3f}")


def example_deletion_analysis():
    """Analyze effects of deleting regulatory elements."""
    print("\n=== DELETION ANALYSIS ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Delete different sized regions
    deletions = [
        ('1bp deletion', 'chr8:127735000-127735001', 'N'),
        ('10bp deletion', 'chr8:127735000-127735010', 'N' * 10),
        ('100bp deletion', 'chr8:127735000-127735100', 'N' * 100),
        ('1kb deletion', 'chr8:127735000-127736000', 'N' * 1000)
    ]
    
    for name, region, replacement in deletions:
        result = oracle.predict_region_replacement(
            genomic_region=region,
            seq=replacement,  # Replace with N's to simulate deletion
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        
        mean_signal = result['normalized_scores']['DNase:K562'].mean()
        print(f"{name}: mean signal = {mean_signal:.4f}")


def example_enhancer_engineering():
    """Engineer synthetic enhancers of different strengths."""
    print("\n=== ENHANCER ENGINEERING ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Design enhancers with increasing binding site density
    base_motif = 'CACGTG'  # E-box
    spacer = 'AAAA'
    
    enhancers = []
    for num_sites in [1, 2, 5, 10, 20]:
        # Create enhancer with specified number of binding sites
        enhancer_seq = (base_motif + spacer * 5) * num_sites
        enhancers.append((num_sites, enhancer_seq))
    
    # Test each enhancer
    region = 'chr8:127730000-127732000'  # 2kb region upstream of MYC
    
    for num_sites, seq in enhancers:
        # Pad sequence to match region length
        region_len = 2000
        if len(seq) < region_len:
            seq = seq + 'A' * (region_len - len(seq))
        else:
            seq = seq[:region_len]
        
        result = oracle.predict_region_replacement(
            genomic_region=region,
            seq=seq,
            assay_ids=['DNase:K562', 'ChIP-seq_H3K27ac:K562'],
            genome=str(genome_path)
        )
        
        dnase_signal = result['normalized_scores']['DNase:K562'].mean()
        h3k27ac_signal = result['normalized_scores']['ChIP-seq_H3K27ac:K562'].mean()
        
        print(f"{num_sites} E-boxes: DNase={dnase_signal:.3f}, H3K27ac={h3k27ac_signal:.3f}")


def example_microsatellite_analysis():
    """Analyze effects of microsatellite repeat variations."""
    print("\n=== MICROSATELLITE ANALYSIS ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Test different repeat lengths
    position = 'chr8:127735000'
    repeat_unit = 'CAG'  # Trinucleotide repeat
    
    for num_repeats in [3, 10, 20, 30, 50]:
        repeat_seq = repeat_unit * num_repeats
        
        result = oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=repeat_seq,
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        
        max_signal = result['normalized_scores']['DNase:K562'].max()
        print(f"{num_repeats} repeats ({len(repeat_seq)}bp): max signal = {max_signal:.3f}")


def example_indel_effects():
    """Analyze small insertions and deletions."""
    print("\n=== INDEL EFFECTS ===")
    
    genome_path = get_genome('hg38')
    oracle = chorus.create_oracle('enformer', 
                                 use_environment=True,
                                 reference_fasta=str(genome_path))
    oracle.load_pretrained_model()
    
    # Common indel sizes
    position = 'chr8:127735434'
    
    # Insertions
    print("Insertions:")
    for size in [1, 2, 3, 4, 5, 10]:
        insert_seq = 'A' * size
        result = oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=insert_seq,
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        signal = result['normalized_scores']['DNase:K562'].mean()
        print(f"  +{size}bp: mean signal = {signal:.4f}")
    
    # Deletions (simulated by replacing with N's)
    print("\nDeletions:")
    for size in [1, 2, 3, 4, 5, 10]:
        region = f'chr8:127735434-{127735434 + size}'
        delete_seq = 'N' * size
        result = oracle.predict_region_replacement(
            genomic_region=region,
            seq=delete_seq,
            assay_ids=['DNase:K562'],
            genome=str(genome_path)
        )
        signal = result['normalized_scores']['DNase:K562'].mean()
        print(f"  -{size}bp: mean signal = {signal:.4f}")


def main():
    """Run all examples."""
    print("Flexible Region Manipulation Examples")
    print("====================================")
    
    # Check environment
    from chorus.core.environment import EnvironmentManager
    env_manager = EnvironmentManager()
    
    if not env_manager.environment_exists('enformer'):
        print("\nERROR: Enformer environment not set up.")
        print("Please run: chorus setup --oracle enformer")
        return
    
    try:
        # Run examples
        example_snp_simulation()
        example_motif_insertion()
        example_deletion_analysis()
        example_enhancer_engineering()
        example_microsatellite_analysis()
        example_indel_effects()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()