#!/usr/bin/env python
"""Quick test script to verify Chorus installation."""

import sys
print("Testing Chorus installation...\n")

# Test imports
try:
    import chorus
    print(f"✓ Chorus imported successfully (version {chorus.__version__})")
except ImportError as e:
    print(f"✗ Failed to import chorus: {e}")
    sys.exit(1)

# Test core components
try:
    import numpy as np
    import pandas as pd
    
    # Test Track creation
    track_data = pd.DataFrame({
        'chrom': ['chr1'] * 5,
        'start': [0, 100, 200, 300, 400],
        'end': [100, 200, 300, 400, 500],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    track = chorus.Track(
        name="test_track",
        assay_type="DNase",
        cell_type="K562",
        data=track_data
    )
    print("✓ Track creation works")
    
    # Test normalization
    norm_track = track.normalize('zscore')
    print("✓ Track normalization works")
    
except Exception as e:
    print(f"✗ Error testing Track: {e}")

# Test oracle creation
try:
    # First check if we can create an oracle with environment isolation
    print("\nTesting oracle system:")
    
    # Check environment manager
    from chorus.core.environment import EnvironmentManager
    manager = EnvironmentManager()
    envs = manager.list_environments()
    print(f"✓ Environment manager works")
    print(f"  - Available environments: {len(envs)}")
    
    # Try to create oracle with environment (won't fail even without deps)
    try:
        oracle = chorus.create_oracle('enformer', use_environment=True)
        print("✓ Oracle creation with environment isolation works")
    except Exception as e:
        print(f"! Oracle with environment isolation not available: {e}")
        print("  Run 'chorus setup --oracle enformer' to set up the environment")
    
except Exception as e:
    print(f"✗ Error with oracle system: {e}")

# Test sequence utilities
try:
    seq = "ATCGATCGATCG"
    gc = chorus.get_gc_content(seq)
    rev_comp = chorus.reverse_complement(seq)
    print(f"✓ Sequence utilities work")
    print(f"  - GC content of {seq}: {gc:.2%}")
    print(f"  - Reverse complement: {rev_comp}")
except Exception as e:
    print(f"✗ Error with sequence utilities: {e}")

# Test variant parsing
try:
    # Create a temporary VCF for testing
    import tempfile
    import os
    
    vcf_content = """##fileformat=VCFv4.3
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\trs123\tA\tG\t30\tPASS\t.
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
        f.write(vcf_content)
        vcf_file = f.name
    
    variants = chorus.parse_vcf(vcf_file)
    os.unlink(vcf_file)
    print(f"✓ VCF parsing works ({len(variants)} variant loaded)")
except Exception as e:
    print(f"✗ Error with VCF parsing: {e}")

# Check for optional dependencies
print("\nOptional dependencies:")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} available")
except ImportError:
    print("✗ TensorFlow not available (needed for Enformer)")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} available")
except ImportError:
    print("✗ PyTorch not available (needed for other models)")

try:
    import kipoiseq
    print("✓ Kipoiseq available")
except ImportError:
    print("✗ Kipoiseq not available (optional for Enformer)")

print("\n✅ Basic installation test complete!")
print("\nNext steps:")
print("1. Try the example notebooks in examples/")
print("2. To use Enformer, you'll need to load the model (large download)")
print("3. For genomic operations, you'll need reference genome files")