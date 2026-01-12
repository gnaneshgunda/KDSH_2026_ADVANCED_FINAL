# RAG System Improvements for Claim Verification

## Issues Identified from Error Analysis

### 1. **Chunking Issues**
- **Problem**: Chunk size too small (120 words) → missing context
- **Problem**: No metadata stored in chunks (character, location, timeline)
- **Solution**: 
  - Increased chunk size to 400 words with 80-word overlap
  - Added metadata extraction during chunking:
    - Characters mentioned (capitalized names)
    - Temporal markers (years, seasons, time periods, life stages)
    - Location references (place names)
    - Dialogue detection (has quotes)

### 2. **Retrieval Precision Issues**
- **Problem**: Not using character-specific filtering effectively
- **Problem**: No reranking → poor relevance
- **Solution**:
  - Multi-stage retrieval pipeline:
    1. Metadata filtering (character mentions)
    2. Semantic ranking (cosine similarity)
    3. Reranking (NVIDIA rerank API)
    4. Context expansion (±1 neighboring chunks)
  - Fuzzy character matching (substring matching)

### 3. **Weak Rationale Generation**
- **Problem**: Generic rationales like "Consistent. Verified 4/5 claims"
- **Problem**: No evidence quotes in output
- **Solution**:
  - Detailed contradiction rationales with:
    - Specific claim that failed
    - Verification explanation
    - Evidence snippets (first 150 chars)
  - Detailed support rationales with:
    - List of verified claims (up to 4)
    - Verification explanations for each
    - Count of additional verified claims

### 4. **Claim Verification Prompt Issues**
- **Problem**: Too many false positives (marking NOT_MENTIONED as SUPPORTED)
- **Problem**: Confusing inferences with explicit statements
- **Solution**:
  - Stricter verification rules:
    - SUPPORTED: Only explicit statements or direct synonyms
    - CONTRADICTED: Only explicit contradictions or logical impossibilities
    - NOT_MENTIONED: Thematic consistency ≠ support, inferences ≠ support
  - Examples in prompt to clarify edge cases
  - Requirement to quote specific evidence

## Key Changes by File

### config.py
```python
# Changed chunk size from 120 to 400 words
DEFAULT_CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
```

### chunker.py
- Added `_extract_metadata()` method
- Returns list of dicts with `text`, `start_pos`, `end_pos`, `metadata`
- Metadata includes: characters, temporal, locations, has_dialogue

### models.py
- Added fields to ChunkMetadata:
  - `locations: List[str]`
  - `has_dialogue: bool`

### index_manager.py
- Updated `_build_book_index()` to use new chunker output
- Stores metadata in ChunkMetadata objects
- Metadata persisted in .pkl files

### retriever.py (HybridRetriever)
- Added `_filter_by_metadata()` for character filtering
- Added `use_rerank` parameter (default True)
- Reranking with NVIDIA API (fallback to cosine)
- Reduced context_window from 2 to 1 (more focused)

### claim_verifier.py
- Completely rewritten prompt with:
  - Clear SUPPORTED/CONTRADICTED/NOT_MENTIONED definitions
  - Examples of edge cases
  - Explicit instruction: "Only EXPLICIT statements count"
  - Requirement to quote evidence in rationale
- Better error handling with detailed fallback messages

### pipeline.py
- Added `_build_contradiction_rationale()` method
- Added `_build_support_rationale()` method
- Updated `analyze_backstory()` to:
  - Use character filtering in retrieval
  - Enable reranking
  - Generate detailed rationales with evidence

## Expected Improvements

### Accuracy
- **Fewer false positives**: Stricter verification rules
- **Fewer false negatives**: Better retrieval with metadata filtering
- **Better contradiction detection**: Explicit evidence requirements

### Rationale Quality
- **More specific**: Quotes from evidence
- **More actionable**: Shows which claims failed/passed
- **More transparent**: Shows verification reasoning

### Retrieval Quality
- **Higher precision**: Character-specific filtering
- **Better ranking**: Reranking with NVIDIA API
- **More context**: Neighboring chunks included

## How to Test

1. **Delete existing .pkl files** to rebuild with new metadata:
   ```bash
   del db\*.pkl
   ```

2. **Run pipeline**:
   ```bash
   python pipeline.py
   ```

3. **Check improvements**:
   - Rationales should be longer and more detailed
   - Should see evidence quotes in rationales
   - Confidence scores should be more calibrated
   - Fewer "unknown" verdicts

## Common Error Patterns Fixed

### Error Pattern 1: "Thematic consistency mistaken for support"
- **Example**: Claim "He was sad" + Evidence "He was crying" → Was SUPPORTED, now NOT_MENTIONED
- **Fix**: Prompt explicitly states "inferences are NOT support"

### Error Pattern 2: "Missing character context"
- **Example**: Claim about "Thalcave" retrieves chunks about other characters
- **Fix**: Character filtering in retrieval

### Error Pattern 3: "Small chunks missing context"
- **Example**: Chunk cuts off mid-sentence, loses meaning
- **Fix**: Increased chunk size to 400 words

### Error Pattern 4: "Generic rationales"
- **Example**: "Consistent. Verified 4/5 claims"
- **Fix**: Shows which claims, with evidence quotes

## Next Steps for Further Improvement

1. **Add temporal reasoning**: Filter chunks by time period mentioned in claim
2. **Add location reasoning**: Filter chunks by location mentioned in claim
3. **Multi-hop reasoning**: Connect related chunks for complex claims
4. **Confidence calibration**: Tune confidence thresholds based on validation set
5. **Claim importance weighting**: Weight critical claims higher than minor details
