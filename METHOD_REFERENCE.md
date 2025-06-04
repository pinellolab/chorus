# Chorus Method Reference

## Quick Reference Table

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `predict()` | Sequence or coordinates | Track predictions | Basic prediction |
| `predict_region_replacement()` | Region + replacement seq | Modified predictions | Test enhancers/promoters |
| `predict_region_insertion_at()` | Position + insert seq | Modified predictions | Test insertions |
| `predict_variant_effect()` | Position + alleles | Effect sizes | SNP/variant analysis |
| `analyze_gene_expression()` | Predictions + gene | Expression levels | Gene-specific analysis |

## Detailed Method Specifications

### predict()

**Purpose**: Get regulatory predictions for a DNA sequence or genomic region.

**Inputs**:
```python
# Option 1: DNA sequence
seq = "ACGT" * 98304  # Must be exactly 393,216 bp for Enformer
predictions = oracle.predict(seq, ['ENCFF413AHU'])

# Option 2: Genomic coordinates (requires reference_fasta)
predictions = oracle.predict(('chrX', 48777634, 48790694), ['ENCFF413AHU'])
```

**Track IDs (Enformer/Borzoi)**:
- ENCODE format: `'ENCFF413AHU'` (specific experiment)
- CAGE format: `'CNhs11250'` (specific library)
- Description: `'DNase:K562'`, `'H3K27ac:HepG2'`

**Output Structure**:
```python
{
    'ENCFF413AHU': np.array([...]),  # Shape: (896,) for Enformer
    'CNhs11250': np.array([...])      # Each bin = 128 bp
}
```

**Key Points**:
- Each prediction value represents signal in a 128 bp bin
- Values are model's raw outputs (not normalized)
- For Enformer: 896 bins × 128 bp = 114,688 bp output

---

### predict_region_replacement()

**Purpose**: Replace a genomic region with custom sequence and see the effects.

**Input Requirements**:
- `genomic_region`: Target region to replace
- `seq`: Must be EXACTLY same length as region
- Region can be any size from 1 bp to full context

**Logic Flow**:
```
1. Region: chr11:5247400-5247600 (200 bp)
2. Center: 5247500
3. Context window: [center - 196608, center + 196608] = 393,216 bp
4. Extract context from genome
5. Replace bases 5247400-5247600 within context
6. Predict on full modified context
7. Return predictions for output window
```

**Example - Test Enhancer**:
```python
# Replace 200bp with GATA motif repeats
enhancer = "GATA" * 50  # Exactly 200 bp
results = oracle.predict_region_replacement(
    "chr11:5247400-5247600",  # 200 bp region
    enhancer,
    ['ENCFF413AHU', 'CNhs11250']
)

# Compare with wild-type
wt = oracle.predict(('chr11', 5247000, 5248000), tracks)
change_dnase = results['raw_predictions']['ENCFF413AHU'] - wt['ENCFF413AHU']
```

**Common Use Cases**:
- Test synthetic enhancers/silencers
- Remove regulatory elements (replace with 'N's)
- Swap promoters between genes
- Test TFBS clusters

---

### predict_region_insertion_at()

**Purpose**: Insert sequence at a specific position without removing existing sequence.

**Input Requirements**:
- `genomic_position`: Exact insertion point
- `seq`: Any length that fits within context
- Maximum insert: ~390 kb (leaving 3 kb flanks)

**Logic Flow**:
```
1. Position: chr11:5247500
2. Insert size: 200 bp
3. Required flanks: (393216 - 200) / 2 = 196508 bp each side
4. Left flank: [5247500 - 196508, 5247500]
5. Right flank: [5247500, 5247500 + 196508]
6. Final: left_flank + insert + right_flank
```

**Example - Insert Enhancer**:
```python
# Insert enhancer at TSS
results = oracle.predict_region_insertion_at(
    "chr11:5247500",  # Insertion point
    "GATA" * 50,      # 200 bp insert
    ['CNhs11250']     # CAGE to measure expression
)
```

**Comparison with Replacement**:
- Replacement: Keeps genome length same, changes existing sequence
- Insertion: Adds new sequence, shifts downstream regions

---

### predict_variant_effect()

**Purpose**: Test effects of genetic variants (SNPs, small indels).

**Input Structure**:
```python
results = oracle.predict_variant_effect(
    genomic_region="chr11:5247000-5248000",  # Context region
    variant_position="chr11:5247500",        # Variant location
    alleles=['C', 'A', 'G', 'T'],           # Reference first
    assay_ids=['ENCFF413AHU']
)
```

**Output Structure**:
```python
{
    'predictions': {
        'reference': {'ENCFF413AHU': array(...)},  # C allele
        'alt_1': {'ENCFF413AHU': array(...)},      # A allele
        'alt_2': {'ENCFF413AHU': array(...)},      # G allele
        'alt_3': {'ENCFF413AHU': array(...)}       # T allele
    },
    'effect_sizes': {
        'alt_1': {'ENCFF413AHU': array(...)},  # A - C
        'alt_2': {'ENCFF413AHU': array(...)},  # G - C
        'alt_3': {'ENCFF413AHU': array(...)}   # T - C
    },
    'variant_info': {
        'position': 'chr11:5247500',
        'ref': 'C',
        'alts': ['A', 'G', 'T']
    }
}
```

**Analyzing Results**:
```python
# Get mean effect across output window
mean_effect = np.mean(results['effect_sizes']['alt_1']['ENCFF413AHU'])

# Find maximum effect position
max_idx = np.argmax(np.abs(results['effect_sizes']['alt_1']['ENCFF413AHU']))
max_effect_pos = start + max_idx * 128  # Convert to genomic coordinate

# Check all alleles
for i, alt in enumerate(['A', 'G', 'T']):
    effect = results['effect_sizes'][f'alt_{i+1}']['ENCFF413AHU']
    print(f"C→{alt}: mean effect = {np.mean(effect):.4f}")
```

---

### analyze_gene_expression() [Enformer-specific]

**Purpose**: Analyze predicted gene expression using CAGE signal at TSS.

**Requirements**:
- Gene annotation file (GTF)
- CAGE track predictions
- Gene must have annotated TSS in region

**Example**:
```python
expr_analysis = oracle.analyze_gene_expression(
    predictions=predictions,      # From predict()
    gene_name='GATA1',
    chrom='chrX',
    start=48777634,
    end=48790694,
    gtf_file='gencode.gtf',
    cage_track_ids=['CNhs11250']  # CAGE tracks only
)

# Returns
{
    'mean_expression': {'CNhs11250': 120.5},  # Average at TSS
    'max_expression': {'CNhs11250': 150.2},   # Peak at TSS
    'tss_positions': [48786540, 48786562],    # TSS locations
    'n_tss': 2                                 # Number of TSS
}
```

---

### save_predictions_as_bedgraph()

**Purpose**: Export predictions for genome browser visualization.

**Enformer Coordinate Mapping**:
- Automatically handles input → output window mapping
- No need to manually calculate output coordinates

**Example**:
```python
# For wild-type region
files = oracle.save_predictions_as_bedgraph(
    predictions=predictions,
    chrom='chrX',
    start=48777634,  # Original region start
    end=48790694,    # Original region end
    output_dir='tracks',
    prefix='gata1_wt'
)

# Produces files like:
# tracks/gata1_wt_ENCFF413AHU.bedgraph
# tracks/gata1_wt_CNhs11250.bedgraph
```

**Loading in IGV/UCSC**:
1. Open genome browser
2. File → Load from File
3. Select .bedgraph files
4. Adjust track height and color

---

## Coordinate Systems Explained

### 1. User Input (1-based, inclusive)
```
"chr1:1000-2000" means:
- Includes base 1000
- Includes base 2000  
- Total: 1001 bases
```

### 2. Internal (0-based, half-open)
```python
# After parsing "chr1:1000-2000"
start = 999   # 0-based
end = 2000    # exclusive
# Slice: seq[999:2000] = 1001 bases
```

### 3. Enformer-Specific Mapping
```
Input:  [--------393,216 bp--------]
         ↑                        ↑
         139,264 bp offset

Output:     [---114,688 bp---]
            ↑                ↑
            896 bins × 128 bp
```

---

## Common Patterns

### Compare Wild-Type vs Modified
```python
# 1. Get wild-type
wt = oracle.predict(region, tracks)

# 2. Test modification  
modified = oracle.predict_region_replacement(sub_region, new_seq, tracks)

# 3. Compare
for track in tracks:
    wt_mean = np.mean(wt[track])
    mod_mean = np.mean(modified['raw_predictions'][track])
    change = ((mod_mean - wt_mean) / wt_mean) * 100
    print(f"{track}: {change:+.1f}% change")
```

### Scan for Optimal Insertion Site
```python
positions = range(start, end, 100)  # Every 100 bp
effects = []

for pos in positions:
    result = oracle.predict_region_insertion_at(
        f"chr1:{pos}",
        enhancer_seq,
        ['CAGE:K562']
    )
    effect = np.mean(result['raw_predictions']['CAGE:K562'])
    effects.append(effect)

best_pos = positions[np.argmax(effects)]
```

### Test All Variants in a Region
```python
# Get reference sequence
ref_seq = extract_sequence(f"chr1:{start}-{end}", genome)

for i, base in enumerate(ref_seq):
    pos = start + i
    alts = [b for b in 'ACGT' if b != base]
    
    result = oracle.predict_variant_effect(
        f"chr1:{start}-{end}",
        f"chr1:{pos}",
        [base] + alts,
        tracks
    )
    
    # Store max effect
    max_effect = max(
        np.max(np.abs(result['effect_sizes'][f'alt_{j+1}'][track]))
        for j in range(len(alts))
        for track in tracks
    )
```

---

## Troubleshooting

### "Reference allele mismatch"
- Check you're using correct genome version (hg38 vs hg19)
- Verify coordinate system (1-based for genomic regions)
- Use `extract_sequence()` to check actual reference

### "Sequence length mismatch"  
- Enformer requires exactly 393,216 bp
- Replacement sequence must match region length exactly
- Use padding with 'N's if needed

### "Track not found"
```python
# List all available tracks
oracle.list_assay_types()
oracle.list_cell_types()

# Get track info
info = oracle.get_track_info("ENCFF413AHU")
```

### Memory Issues
- Reduce number of tracks predicted at once
- Process variants in batches
- Use CPU-only mode if GPU memory limited