# Chorus Implementation Guide

## Architecture Overview

Chorus uses a modular architecture with the following key components:

```
chorus/
├── core/
│   ├── base.py          # Abstract base class for all oracles
│   ├── track.py         # Track data structure
│   ├── exceptions.py    # Custom exceptions
│   └── environment/     # Environment management
├── oracles/
│   ├── enformer.py      # Enformer implementation
│   ├── borzoi.py        # Placeholder
│   ├── chrombpnet.py    # Placeholder
│   └── sei.py           # Placeholder
├── utils/
│   ├── sequence.py      # DNA sequence utilities
│   ├── genome.py        # Genome management
│   ├── annotations.py   # Gene annotation handling
│   ├── visualization.py # Plotting functions
│   └── normalization.py # Signal normalization
└── cli/
    └── main.py          # Command-line interface
```

## Key Design Patterns

### 1. Abstract Base Class Pattern

All oracles inherit from `OracleBase`, which:
- Defines the interface all oracles must implement
- Provides common functionality
- Handles environment management

```python
class OracleBase(ABC):
    @abstractmethod
    def load_pretrained_model(self, weights: str) -> None:
        """Must be implemented by each oracle"""
        pass
    
    @abstractmethod
    def _predict(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Internal prediction method"""
        pass
    
    # Common methods implemented in base class
    def predict(self, input_data, assay_ids):
        # Common validation and processing
        # Calls _predict() which subclasses implement
```

### 2. Environment Isolation

Each oracle runs in its own conda environment to avoid conflicts:

```python
# Environment creation
env_manager = EnvironmentManager()
env_manager.create_environment('enformer', env_file='environments/chorus-enformer.yml')

# Running code in environment
runner = EnvironmentRunner(env_manager)
result = runner.run_in_environment('enformer', func, args, kwargs)
```

### 3. Coordinate System Handling

Chorus handles multiple coordinate systems:

1. **Genomic coordinates** (1-based, inclusive):
   - User-facing: "chr1:1000-2000"
   - Used in method parameters

2. **Python/pysam coordinates** (0-based, half-open):
   - Internal: start=999, end=2000
   - Used with pysam.FastaFile

3. **Relative coordinates** (0-based):
   - Position within extracted sequence
   - Used for variant application

## Method Implementation Details

### Context Window Management

For models with specific input size requirements:

```python
def predict_region_replacement(self, genomic_region, seq, assay_ids):
    # 1. Parse region
    chrom, start, end = self._parse_region(genomic_region)
    region_length = end - start
    
    # 2. Validate replacement matches region
    if len(seq) != region_length:
        raise ValueError("Replacement must match region length")
    
    # 3. Get model's context requirements
    context_size = self._get_context_size()  # e.g., 393,216 for Enformer
    
    # 4. Calculate context window centered on region
    region_center = (start + end) // 2
    context_start = region_center - context_size // 2
    context_end = region_center + context_size // 2
    
    # 5. Extract full context
    context_seq = extract_sequence(f"{chrom}:{context_start}-{context_end}", genome)
    
    # 6. Replace region within context
    replace_start_in_context = start - context_start
    replace_end_in_context = end - context_start
    full_seq = (
        context_seq[:replace_start_in_context] + 
        seq + 
        context_seq[replace_end_in_context:]
    )
    
    # 7. Predict on full context
    predictions = self._predict(full_seq, assay_ids)
```

### Output Window Mapping (Enformer-specific)

Enformer has asymmetric input/output:
- Input: 393,216 bp
- Output: 114,688 bp (centered within input)

```python
def get_output_window_coords(self, region_center: int) -> Tuple[int, int]:
    """Map from input window to output window coordinates."""
    output_size = self.target_length * self.bin_size  # 114,688 bp
    offset = (self.sequence_length - output_size) // 2  # 139,264 bp
    
    input_start = region_center - self.sequence_length // 2
    output_start = input_start + offset
    output_end = output_start + output_size
    
    return output_start, output_end
```

### Track Identifier Resolution

Different oracles use different track naming:

```python
def _get_assay_indices(self, assay_ids: List[str]) -> List[int]:
    """Convert track IDs to model indices."""
    indices = []
    
    for assay_id in assay_ids:
        if assay_id in self._track_dict:
            # Direct match (e.g., ENCFF413AHU)
            indices.append(self._track_dict[assay_id])
        elif ':' in assay_id:
            # Description format (e.g., DNase:K562)
            matches = self._find_tracks_by_description(assay_id)
            indices.extend(matches)
        else:
            raise InvalidAssayError(f"Unknown track: {assay_id}")
    
    return indices
```

### Variant Effect Calculation

```python
def predict_variant_effect(self, genomic_region, variant_position, alleles, assay_ids):
    # 1. Extract reference sequence
    ref_seq = extract_sequence(genomic_region, genome)
    
    # 2. Calculate variant position within region
    region_start = parse_start(genomic_region)
    relative_pos = variant_position - region_start
    
    # 3. Validate reference allele
    ref_allele = alleles[0]
    if ref_seq[relative_pos] != ref_allele:
        raise ValueError("Reference allele mismatch")
    
    # 4. Create sequences for each allele
    sequences = {}
    sequences['reference'] = ref_seq
    
    for i, alt in enumerate(alleles[1:]):
        alt_seq = apply_variant(ref_seq, relative_pos, ref_allele, alt)
        sequences[f'alt_{i+1}'] = alt_seq
    
    # 5. Predict for each sequence
    all_predictions = {}
    for name, seq in sequences.items():
        predictions = self._predict(seq, assay_ids)
        all_predictions[name] = predictions
    
    # 6. Calculate effects
    effect_sizes = {}
    for alt_name in ['alt_1', 'alt_2', ...]:
        effect_sizes[alt_name] = {}
        for assay in assay_ids:
            effect = all_predictions[alt_name][assay] - all_predictions['reference'][assay]
            effect_sizes[alt_name][assay] = effect
```

## File Format Specifications

### BedGraph Format
```
track name="DNase:K562" type=bedGraph
chr1    0       128     0.5432
chr1    128     256     0.6721
chr1    256     384     0.4839
```

### Track Metadata (Enformer)
```json
{
  "ENCFF413AHU": {
    "description": "DNASE:K562",
    "cell_type": "K562",
    "assay_type": "DNASE",
    "experiment": "ENCSR000EOT"
  }
}
```

## Performance Considerations

### Memory Management
- Enformer model: ~1-2 GB
- Predictions: ~20 MB per 393kb sequence
- Use batch processing for multiple sequences

### Speed Optimizations
1. **Environment caching**: Environments persist between calls
2. **Model caching**: Models loaded once per session
3. **Genome indexing**: FASTA files indexed for fast random access

### Batch Processing (Future)
```python
# Planned implementation
predictions = oracle.predict_batch(
    sequences=[seq1, seq2, seq3],
    assay_ids=['ENCFF413AHU'],
    batch_size=4
)
```

## Adding New Oracles

To implement a new oracle:

1. **Create oracle class**:
```python
# chorus/oracles/newmodel.py
class NewModelOracle(OracleBase):
    def __init__(self, use_environment=True, reference_fasta=None):
        self.oracle_name = 'newmodel'
        super().__init__(use_environment)
        self.sequence_length = 1000  # Model-specific
        
    def load_pretrained_model(self, weights=None):
        # Load model architecture and weights
        
    def _predict(self, seq, assay_ids):
        # Run model prediction
        
    def _get_context_size(self):
        return self.sequence_length
```

2. **Create environment file**:
```yaml
# environments/chorus-newmodel.yml
name: chorus-newmodel
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.9
  - pytorch=2.0
  - model-specific-deps
```

3. **Register in factory**:
```python
# chorus/__init__.py
ORACLE_REGISTRY = {
    'enformer': EnformerOracle,
    'newmodel': NewModelOracle,
}
```

## Testing

### Unit Tests
```python
# tests/test_oracles.py
def test_prediction_shape():
    oracle = EnformerOracle()
    oracle.load_pretrained_model()
    
    seq = 'ACGT' * 98304  # 393,216 bp
    pred = oracle.predict(seq, ['ENCFF413AHU'])
    
    assert pred['ENCFF413AHU'].shape == (896,)
```

### Integration Tests
```python
def test_variant_effect_workflow():
    # Test full workflow from genomic coords to effect sizes
    oracle = create_oracle('enformer', reference_fasta='hg38.fa')
    oracle.load_pretrained_model()
    
    results = oracle.predict_variant_effect(
        'chr1:1000000-1001000',
        'chr1:1000500',
        ['A', 'T'],
        ['DNase:K562']
    )
    
    assert 'effect_sizes' in results
    assert 'alt_1' in results['effect_sizes']
```

## Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show:
# - Environment activation
# - Sequence extraction
# - Model loading steps
# - Prediction shapes
```

### Common Issues

1. **Memory errors**: Reduce batch size or use CPU-only mode
2. **Environment conflicts**: Delete and recreate environment
3. **Coordinate mismatches**: Check 0-based vs 1-based
4. **Track not found**: Use `oracle.list_assay_types()` to see valid IDs

### Validation Functions
```python
# Check sequence
oracle._validate_sequence(seq)  # Length and content

# Check coordinates
oracle._validate_region("chr1:1000-2000")  # Format

# Check track IDs
oracle._validate_assay_ids(['ENCFF413AHU'])  # Valid IDs
```