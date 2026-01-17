# Advanced Narrative Consistency RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system for analyzing narrative consistency in character backstories. This system uses advanced NLP techniques, semantic chunking, claim extraction, and multi-stage verification to detect contradictions between character backstories and source narratives.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Output Format](#output-format)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

---

## Overview

This RAG pipeline verifies whether character backstories are consistent with source narrative texts. It extracts atomic factual claims from backstories, retrieves relevant narrative evidence, and uses an LLM to verify each claim's consistency.

**Key Innovation**: Single-strike contradiction detection with detailed rationale generation using multi-stage retrieval and strict verification rules.

### What the Pipeline Does

1. **Loads narrative texts** from book files (`.txt` format)
2. **Chunks texts** into semantic segments with NLP-enriched metadata
3. **Creates embeddings** and builds searchable indices (cached per book)
4. **Processes CSV records** containing character backstories
5. **Extracts atomic claims** from each backstory using LLM
6. **Retrieves relevant evidence** using hybrid retrieval (metadata filtering + semantic search + reranking)
7. **Verifies each claim** against narrative evidence
8. **Generates verdicts** with confidence scores and detailed rationales
9. **Outputs results** to CSV format

---

## ✨ Features

### Core Capabilities

- **Semantic Chunking**: NLP-aware text segmentation with overlap
- **Rich Metadata Extraction**: Characters, locations, temporal markers, dialogue detection
- **Hybrid Retrieval**: Character-based filtering + cosine similarity + NVIDIA reranking
- **Claim Extraction**: Atomic, verifiable facts extracted from backstories
- **Strict Verification**: Explicit evidence required (no inferences)
- **Single-Strike Logic**: One contradiction → inconsistent verdict
- **Detailed Rationales**: Evidence quotes and claim-level explanations
- **Per-Book Caching**: Efficient `.pkl` storage for processed books
- **Fallback Support**: HuggingFace fallback when NVIDIA API unavailable

### Advanced Features

- **Context Window Expansion**: Includes neighboring chunks for better context
- **Entity-Focused Retrieval**: Filters by character mentions
- **Confidence Scoring**: Dynamic confidence based on verification results
- **Optional Pathway Integration**: Real-time indexing for dynamic datasets
- **Incremental Processing**: CSV row-by-row processing with progress logging

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INPUT DATA                         │
│  ┌──────────────┐          ┌─────────────────┐     │
│  │ Books (.txt) │          │ Backstories CSV │     │
│  └──────┬───────┘          └────────┬────────┘     │
└─────────┼──────────────────────────┼───────────────┘
          │                          │
          ▼                          ▼
┌─────────────────┐        ┌──────────────────┐
│ Index Manager   │        │ Claim Extractor  │
│ - Chunker       │        │ (NLP + LLM)      │
│ - Embeddings    │        └────────┬─────────┘
│ - Cache (.pkl)  │                 │
└────────┬────────┘                 │
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────┐
│        Hybrid Retriever                 │
│  1. Metadata Filtering (characters)     │
│  2. Semantic Ranking (cosine)           │
│  3. Reranking (NVIDIA API)              │
│  4. Context Expansion                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Claim Verifier│
         │ (LLM)         │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Pipeline Logic│
         │ - Single-Strike│
         │ - Confidence   │
         │ - Rationale    │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ results.csv   │
         └───────────────┘
```

### Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| **Pipeline Orchestrator** | `pipeline.py` | Main execution flow and verdict logic |
| **Configuration** | `config.py` | Environment setup, API keys, parameters |
| **NVIDIA Client** | `nvidia_client.py` | Embeddings, chat, reranking with fallback |
| **Chunker** | `chunker.py` | NLP-aware text segmentation |
| **Claim Extractor** | `claim_extractor.py` | Atomic claim extraction from backstories |
| **Claim Verifier** | `claim_verifier.py` | Evidence-based claim verification |
| **Retriever** | `retriever.py` | Multi-stage hybrid retrieval |
| **Index Manager** | `index_manager.py` | Build/cache book indices |
| **Context Builder** | `context_builder.py` | Sentiment and causal analysis |
| **Models** | `models.py` | Data structures (ChunkMetadata, etc.) |

---

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **API Keys**:
  - NVIDIA API key (for embeddings, chat, reranking)
  - HuggingFace token (optional, for fallback)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd KDSH_2026_ADVANCED_FINAL
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `spacy==3.7.5` - NLP processing
- `nltk==3.9.2` - Sentence tokenization
- `numpy==1.26.4` - Numerical operations
- `pandas==2.2.3` - CSV processing
- `scikit-learn==1.4.2` - Cosine similarity
- `sentence-transformers==2.7.0` - Embeddings (fallback)
- `transformers==4.41.2` - HuggingFace models
- `langchain==0.1.20` - RAG framework
- `langgraph==0.0.48` - Graph operations
- `pathway[xpack-llm]==0.post1` - Real-time indexing (optional)
- `requests==2.32.3` - API calls
- `python-dotenv==1.0.1` - Environment variables

### Step 4: Download NLP Models

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# NLTK data will download automatically on first run
```

---

## Configuration

### Step 1: Create Environment File

Copy the template and add your API keys:

```bash
cp .env.template .env
```

### Step 2: Edit `.env` File

```bash
# Required
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
HF_TOKEN=your_huggingface_token_here
```

### Step 3: Organize Your Data

```
project/
├── db/
│   ├── books/
│   │   ├── the count of monte cristo.txt
│   │   └── in search of the castaways.txt
│   ├── train.csv
│   ├── test.csv (optional)
│   └── *.pkl (generated cache files)
├── pipeline.py
└── ...
```

**CSV Format** (`train.csv` or `test.csv`):

| Column | Description | Required |
|--------|-------------|----------|
| `id` | Unique record identifier | Yes |
| `book_name` | Book name (matches `.txt` filename) | Yes |
| `char` | Character name | Yes |
| `content` | Backstory content | Yes* |
| `caption` | Additional story details | No |
| `label` | Ground truth (0=inconsistent, 1=consistent) | No |

*At least one of `content`, `caption`, or similar columns must have text.

### Step 4: Adjust Configuration (Optional)

Edit `config.py` to customize:

```python
# Chunking
DEFAULT_CHUNK_SIZE = 400  # Increase for more context
CHUNK_OVERLAP = 80        # Overlap between chunks

# Retrieval
DEFAULT_TOP_K = 5         # Number of chunks to retrieve
MAX_TOP_K = 20           # Max chunks for fallback
RETRIEVAL_STEP = 3       # Increment for fallback

# LLM
LLM_TEMPERATURE = 0.0    # 0 = deterministic
LLM_MAX_TOKENS = 1024    # Max response length

# Features
USE_PATHWAY = False      # Enable real-time indexing
FALLBACK_ENABLED = True  # Enable retrieval fallback
```

---

## Usage

### Basic Pipeline Execution

```bash
python pipeline.py
```

This will:
1. Load or build book indices (creates `.pkl` files in `./db/`)
2. Read CSV records from `./db/train.csv`
3. Process each record
4. Output results to `./db/results.csv`

### First Run vs. Subsequent Runs

**First Run:**
- Reads all `.txt` files from `./db/books/`
- Chunks text, generates embeddings
- Caches each book as `<book_name>.pkl` in `./db/`
- Processing time: ~2-5 minutes per book

**Subsequent Runs:**
- Loads cached `.pkl` files instantly
- Processing time: <1 second per book

### Rebuilding the Index

If you modify book files or want to rebuild:

**Option 1: Delete cache files**
```bash
# Windows
del db\*.pkl

# macOS/Linux
rm db/*.pkl
```

**Option 2: Use rebuild script** (Windows)
```bash
rebuild_index.bat
```

Then run the pipeline again:
```bash
python pipeline.py
```

---

## Pipeline Components

### 1. Semantic Chunker (`chunker.py`)

**Purpose**: Split narrative text into semantic chunks with NLP metadata

**Key Features:**
- Paragraph-aware splitting
- Sentence boundary preservation
- Overlap for context continuity
- NLP metadata extraction via spaCy

**Metadata Extracted:**
- `characters` - PERSON entities
- `temporal` - DATE, TIME entities
- `locations` - GPE, LOC entities
- `has_dialogue` - Contains quoted speech

**Example:**
```python
chunker = SemanticChunker(max_chunk_size=400)
chunks = chunker.chunk_text(narrative_text)
# Returns: [{'text': '...', 'metadata': {...}, 'start_pos': 0, 'end_pos': 1500}, ...]
```

### 2. Claim Extractor (`claim_extractor.py`)

**Purpose**: Extract atomic, verifiable facts from backstories

**Extraction Rules:**
- Atomic: One fact per claim (no "and", "but", "because")
- De-duplicated: Same fact not extracted twice
- Hard facts prioritized: Identity, location, time, events
- Fluff discarded: Subjective opinions, vague traits

**Example Input:**
```
"In 1995, Elara moved to London after her father, a clockmaker, died in a fire."
```

**Example Output:**
```
- Elara moved to London in 1995
- Elara's father was a clockmaker
- Elara's father died in a fire
```

### 3. Hybrid Retriever (`retriever.py`)

**Purpose**: Multi-stage evidence retrieval

**Stages:**
1. **Metadata Filtering**: Filter chunks mentioning the character
2. **Semantic Ranking**: Cosine similarity with query embedding
3. **Reranking**: NVIDIA rerank API for precision
4. **Context Expansion**: Include neighboring chunks

**Parameters:**
```python
retriever.retrieve(
    query="He was born in Paris",
    character_name="Jean",
    top_k=5,              # Final number of chunks
    context_window=1,     # ±1 neighboring chunks
    use_rerank=True       # NVIDIA reranking
)
```

### 4. Claim Verifier (`claim_verifier.py`)

**Purpose**: Verify claims against evidence using LLM

**Verification Verdicts:**

| Verdict | Rule |
|---------|------|
| `SUPPORTED` | Evidence **explicitly** states the claim or direct synonym |
| `CONTRADICTED` | Evidence **explicitly** contradicts with opposite facts |
| `NOT_MENTIONED` | Evidence is silent, thematic only, or requires inference |

**Critical Rules:**
- Absence of evidence ≠ contradiction
- Thematic consistency ≠ support
- Reasonable inferences ≠ support
- Only **explicit statements** count

**Output:**
```json
{
  "verdict": "SUPPORTED",
  "rationale": "Passage explicitly states 'Jean was born in Paris, 1890'",
  "confidence": 0.95
}
```

### 5. Pipeline Verdict Logic (`pipeline.py`)

**Single-Strike Rule:**
```python
if ANY claim is CONTRADICTED:
    verdict = "0"  # Inconsistent
    confidence = min(0.95, base_confidence)
else:
    verdict = "1"  # Consistent
    confidence = dynamic (0.55 - 0.95)
```

**Confidence Calculation:**
- **Contradicted**: 0.55 + (contradiction_count / total_claims) * 0.45
- **Consistent**: 0.65 + (support_ratio) * 0.30
- **No explicit support**: 0.55 (conservative)

---

## Output Format

### Results CSV (`./db/results.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Record ID from input CSV |
| `verdict` | string | `"0"` (inconsistent) or `"1"` (consistent) |
| `confidence` | float | 0.0 - 1.0 confidence score |
| `rationale` | string | Detailed explanation with evidence quotes |

### Example Output Row

```csv
id,verdict,confidence,rationale
record_123,0,0.9200,"The narrative evidence contradicts the backstory claim.

Claim #3/12: ""He was in London in 1872...""

Verification: The narrative explicitly states he was in Paris during 1872, which directly contradicts the claim.

Relevant narrative passages:
  [1] ""In the spring of 1872, Jean remained in Paris working at his father's shop...""
  [2] ""He did not leave Paris until late 1873, when he departed for London...""
```

### Rationale Types

**For Contradictions:**
```
The narrative evidence contradicts the backstory claim.

Claim #2/8: "She was married in 1850..."

Verification: Narrative explicitly states she married in 1852, not 1850.

Relevant narrative passages:
  [1] "The wedding took place in autumn 1852..."
```

**For Consistent:**
```
Consistent. Verified 5/7 claims against narrative context.

Key verified facts:
  1. "He was born in Paris..." - Evidence explicitly states he was a native Parisian born in 1840...
  2. "His father was a clockmaker..." - Narrative confirms his father operated a clockmaking shop...
  3. "He learned tracking from Thalcave..." - Supported by passage describing their mentorship...
  4. "He served as a guide..." - Documented in chapters 5-7 of the narrative...
  (+1 additional verified claims)
```

---

## Advanced Features

### 1. Pathway Real-Time Indexing

Enable dynamic index updates for streaming data:

```python
# In config.py
USE_PATHWAY = True
```

```bash
# Install pathway
pip install "pathway[xpack-llm]"
```

**Use Case**: When book files are updated frequently, Pathway can re-index in real-time.

### 2. Fallback Retrieval Loop

When a claim returns `UNKNOWN`, the system automatically:
1. Increases `top_k` by `RETRIEVAL_STEP`
2. Retrieves more evidence
3. Re-verifies claim
4. Repeats up to `MAX_RETRIEVAL_ROUNDS` times

```python
# config.py
FALLBACK_ENABLED = True
MAX_RETRIEVAL_ROUNDS = 3
RETRIEVAL_STEP = 3
```

### 3. HuggingFace Fallback

If NVIDIA API is unavailable, the system falls back to:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reranking**: Cosine similarity (no API needed)
- **Chat**: Stub heuristic responses

### 4. Custom Logging

Adjust log verbosity:

```python
# config.py
LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
```

---

## Troubleshooting

### Issue: "No chunks retrieved"

**Cause**: Character name mismatch between backstory and narrative

**Solution:**
1. Check character names in CSV vs. book text
2. View debug logs for metadata filtering
3. Try reducing character filter strictness

```python
# In retriever.py _filter_by_metadata()
# Change line: if any(char_lower in e or e in char_lower for e in entities_lower):
# To: if any(char_lower[:3] in e or e[:3] in char_lower for e in entities_lower):
```

### Issue: "Reranking failed"

**Cause**: NVIDIA rerank API unavailable

**Solution**: System automatically falls back to cosine similarity. No action needed unless you want to disable reranking:

```python
# In pipeline.py analyze_backstory()
evidence = retriever.retrieve(
    claim,
    character_name=character,
    top_k=5,
    context_window=1,
    use_rerank=False  # Disable reranking
)
```

### Issue: "Empty backstory for record"

**Cause**: CSV missing `content` or `caption` columns

**Solution**: Ensure CSV has text in at least one content column. Modify `_extract_backstory_text()` in `pipeline.py` if using custom column names.

### Issue: "Book not found in corpus"

**Cause**: Book name in CSV doesn't match `.txt` filename

**Solution**:
1. Check exact spelling and case
2. Ensure book files are in `./db/books/`
3. Book names are normalized to lowercase

```python
# CSV: book_name = "The Count of Monte Cristo"
# File: ./db/books/the count of monte cristo.txt  
# File: ./db/books/monte_cristo.txt  
```

### Issue: Low accuracy

**Causes & Solutions:**

| Problem | Solution |
|---------|----------|
| Chunks too small | Increase `DEFAULT_CHUNK_SIZE` to 500-600 |
| Not enough evidence | Increase `top_k` to 7-10 in retrieval |
| Weak verification | Check LLM temperature (should be 0.0) |
| Poor claim extraction | Add more entity context in prompts |

---

## ⚡ Performance Tuning

### Speed Optimization

**Problem**: Pipeline too slow

**Solutions:**
1. **Disable reranking**: Saves ~0.5s per claim
   ```python
   use_rerank=False
   ```

2. **Reduce claims**: Limit extracted claims
   ```python
   claims = self.claim_extractor.extract_claims(backstory_text)[:8]  # Reduce from 12
   ```

3. **Reduce top_k**: Fewer chunks to process
   ```python
   top_k=3  # Instead of 5
   ```

4. **Batch processing**: Process CSV in chunks (future improvement)

### Accuracy Optimization

**Problem**: Too many false positives/negatives

**Solutions:**

**For False Positives (predicting consistent when inconsistent):**
- Increase `top_k` to retrieve more evidence
- Increase `context_window` for more narrative context
- Review claim extraction quality
- Strengthen verification prompt (already strict)

**For False Negatives (predicting inconsistent when consistent):**
- Review confidence thresholds
- Check if character filtering is too strict
- Ensure chunk size is adequate (400+ words)
- Verify temporal/location metadata extraction

### Memory Optimization

**Problem**: High memory usage

**Solutions:**
1. Process books one at a time (future improvement)
2. Reduce `EMBEDDING_DIM` (requires NVIDIA API config change)
3. Use smaller spaCy model: `en_core_web_sm` (already default)

---

## Configuration Parameters Reference

### Chunking Parameters

```python
DEFAULT_CHUNK_SIZE = 400    # Words per chunk (recommended: 300-600)
CHUNK_OVERLAP = 80         # Overlap words (recommended: 15-20% of chunk size)
```

### Retrieval Parameters

```python
DEFAULT_TOP_K = 5          # Initial chunks retrieved (recommended: 3-7)
MAX_TOP_K = 20            # Max for fallback (recommended: 15-25)
RETRIEVAL_STEP = 3        # Fallback increment (recommended: 2-5)
```

### Fallback Parameters

```python
MAX_RETRIEVAL_ROUNDS = 3  # Max fallback attempts (recommended: 2-4)
FALLBACK_ENABLED = True   # Enable/disable fallback
```

### LLM Parameters

```python
LLM_TEMPERATURE = 0.0     # 0 = deterministic, 0.2-0.7 = creative
LLM_MAX_TOKENS = 1024     # Max response length
```

### Evidence Parameters

```python
MAX_SUPPORTING_CHUNKS = 6  # Not currently used
MAX_OPPOSING_CHUNKS = 4    # Not currently used
```

---

## Additional Documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start and key improvements
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Detailed changelog and rationale
- **[WHY_IT_WORKS.md](WHY_IT_WORKS.md)** - Technical deep-dive
- **[VISUAL_SUMMARY.md](VISUAL_SUMMARY.md)** - Visual architecture diagrams

---

## Support

For issues, questions, or contributions, please:
1. Check this README and troubleshooting section
2. Review log output for error messages
3. Check configuration settings
4. Verify data format matches expected schema

---

## License

[Add your license information here]

---

## Acknowledgments

- NVIDIA NIM for embeddings and LLM APIs
- spaCy for NLP processing
- HuggingFace for fallback models
- LangChain for RAG framework
