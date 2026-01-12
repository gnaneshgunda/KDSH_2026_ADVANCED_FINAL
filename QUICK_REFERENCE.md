# Quick Reference: RAG System Improvements

## What Changed?

### 1. Chunking (chunker.py)
```python
# BEFORE: Simple text chunks
chunks = ["text1", "text2", ...]

# AFTER: Rich metadata chunks
chunks = [
    {
        'text': "...",
        'start_pos': 0,
        'end_pos': 1500,
        'metadata': {
            'characters': ['Thalcave', 'Glenarvan'],
            'temporal': ['1815', 'childhood', 'morning'],
            'locations': ['Paris', 'London'],
            'has_dialogue': True
        }
    },
    ...
]
```

### 2. Chunk Size (config.py)
```python
# BEFORE
DEFAULT_CHUNK_SIZE = 120  # Too small, missing context

# AFTER
DEFAULT_CHUNK_SIZE = 400  # Better context preservation
```

### 3. Retrieval (retriever.py)
```python
# BEFORE: Simple cosine similarity
evidence = retriever.retrieve(claim, top_k=7)

# AFTER: Multi-stage with metadata filtering
evidence = retriever.retrieve(
    claim,
    character_name="Thalcave",  # Filter by character
    top_k=5,
    context_window=1,
    use_rerank=True  # NVIDIA rerank API
)
```

### 4. Verification (claim_verifier.py)
```python
# BEFORE: Loose rules
# - Thematic consistency → SUPPORTED
# - Inferences → SUPPORTED
# - Absence → CONTRADICTED

# AFTER: Strict rules
# - Only explicit statements → SUPPORTED
# - Only explicit contradictions → CONTRADICTED
# - Everything else → NOT_MENTIONED
```

### 5. Verdict Logic (pipeline.py)
```python
# BEFORE: Aggregate scoring
supported_count / total_claims > 0.5 → consistent

# AFTER: Single-strike rule
if ANY claim CONTRADICTED → verdict = 0
else → verdict = 1
```

### 6. Rationale Generation (pipeline.py)
```python
# BEFORE
"Consistent. Verified 4/5 claims"

# AFTER
"Consistent. Verified 4/5 claims against narrative context.
Key verified facts:
  1. \"He was born in Paris...\" - Evidence explicitly states...
  2. \"His father was a guide...\" - Narrative confirms...
  3. \"He learned tracking...\" - Supported by passage...
  4. \"He met Glenarvan...\" - Documented in chapter..."
```

## How to Use

### Step 1: Rebuild Index
```bash
# Delete old .pkl files
del db\*.pkl

# Or use the rebuild script
rebuild_index.bat
```

### Step 2: Run Pipeline
```bash
python pipeline.py
```

### Step 3: Check Results
```bash
# View results
type db\results.csv

# Compare with ground truth
# Check rationale quality
```

## Key Improvements

### ✅ Better Retrieval
- Character-specific filtering
- Temporal/location metadata
- Reranking for precision

### ✅ Stricter Verification
- Explicit evidence required
- No inferences
- Absence ≠ contradiction

### ✅ Better Rationales
- Evidence quotes
- Specific claims listed
- Verification explanations

### ✅ Single-Strike Logic
- One contradiction → fail
- Prevents aggregate masking

## Configuration Options

### Adjust Chunk Size
```python
# config.py
DEFAULT_CHUNK_SIZE = 400  # Increase for more context
CHUNK_OVERLAP = 80        # Increase for more overlap
```

### Adjust Retrieval
```python
# In pipeline.py analyze_backstory()
evidence = retriever.retrieve(
    claim,
    character_name=character,
    top_k=5,              # Increase for more evidence
    context_window=1,     # Increase for more context
    use_rerank=True       # Disable if API unavailable
)
```

### Adjust Confidence Thresholds
```python
# In pipeline.py _build_support_rationale()
support_ratio = supported_count / total_claims
confidence = 0.65 + (0.30 * support_ratio)  # Adjust formula
```

## Troubleshooting

### Issue: "No chunks retrieved"
**Cause**: Character name mismatch
**Fix**: Check character names in metadata, adjust filtering

### Issue: "Reranking failed"
**Cause**: NVIDIA API unavailable
**Fix**: System falls back to cosine similarity automatically

### Issue: "Rationale too long"
**Cause**: Many verified claims
**Fix**: Truncated to 1000 chars in CSV output

### Issue: "Low confidence on correct predictions"
**Cause**: Few explicit supports
**Fix**: This is expected - system is conservative

## Testing Checklist

- [ ] Old .pkl files deleted
- [ ] New .pkl files created with metadata
- [ ] Results.csv generated
- [ ] Rationales are detailed (>100 chars)
- [ ] Rationales include evidence quotes
- [ ] Confidence scores reasonable (0.5-0.95)
- [ ] Character filtering working (check logs)
- [ ] Reranking working (check logs)

## Performance Expectations

### Accuracy
- **Target**: 85-90% (up from ~70-75%)
- **False Positives**: Should decrease significantly
- **False Negatives**: Should decrease moderately

### Speed
- **Slower**: Reranking adds ~0.5s per claim
- **Mitigation**: Can disable reranking if needed

### Rationale Quality
- **Length**: 200-800 chars (up from 50-150)
- **Detail**: Includes evidence quotes
- **Actionability**: Shows which claims verified
