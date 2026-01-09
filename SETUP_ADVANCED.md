# Advanced KDSH 2026 RAG - Complete Setup

## Architecture Overview

```
Input: CSV with book names + character backstories
    ↓
[Dependency Parsing]
  - Intelligent chunking using edge density
  - Build dependency graphs per chunk
  - Extract entities + relationships
    ↓
[Context Vector Construction]
  - Temporal markers (past/present/future)
  - Sentiment analysis (positive/negative)
  - Causal indicators (cause/effect chains)
    ↓
[NVIDIA Embeddings]
  - Embed narrative chunks + backstory claims
  - Store both embeddings + context vectors
    ↓
[Semantic Negation]
  - Generate semantic opposites of backstory claims
  - Find "opposing" chunks that contradict
    ↓
[Graph-RAG Multi-hop]
  - Build knowledge graph of narrative chunks
  - Find related chunks via multi-hop paths
  - Support causal reasoning chains
    ↓
[Supporting + Opposing Retrieval]
  - Get chunks supporting the backstory
  - Get chunks opposing/contradicting it
    ↓
[LLM Reasoning]
  - Feed supporting + opposing evidence to LLM
  - LLM judges consistency with full context
    ↓
Output: Consistency prediction + confidence + reasoning
```

## Prerequisites

```bash
# System packages
apt-get install build-essential python3-dev  # Linux

# Python packages (see requirements_advanced.txt)
pip install -r requirements_advanced.txt

# spaCy language model (important!)
python -m spacy download en_core_web_md

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## Setup Steps

### 1. Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scriptsctivate.bat  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements_advanced.txt
```

### 3. Download spaCy Model (Required!)

```bash
python -m spacy download en_core_web_md
```

### 4. NVIDIA Configuration

```bash
cp .env.template .env

# Edit .env with your NVIDIA API key
# Option A (Cloud): NVIDIA_API_KEY=nvapi-xxxxx
# Option B (Local): NVIDIA_API_KEY=local, NVIDIA_BASE_URL=http://localhost:8000/v1

# Verify
python test_nvidia_advanced.py
```

### 5. Prepare Data

```bash
mkdir -p books
# Copy book .txt files
cp /your/books/*.txt books/

# Copy training CSV
cp training_data.csv .
```

### 6. Run

```bash
python rag_advanced.py
# Output: results_advanced.csv
```

## Key Components

### 1. DependencyChunker
- Uses spaCy dependency parsing
- Chunks text based on syntactic boundaries + size limits
- Builds NetworkX dependency graphs per chunk
- Extracts entities & relationships

### 2. ContextVectorBuilder
Creates multi-dimensional context:
- **Temporal**: past/present/future markers
- **Emotional**: sentiment (positive/negative)
- **Causal**: cause/effect relationships

Concatenates to embedding: [original_embedding | sentiment | temporal | causal]

### 3. SemanticNegationFinder
- Uses LLM to generate semantic opposites
- Finds "contradicting" chunks in narrative
- Enables detection of logical inconsistencies

### 4. GraphRAG
- Builds narrative knowledge graph
- Nodes: chunks, Edges: semantic similarity
- Multi-hop search for related contexts
- Enables causal reasoning chains

### 5. AdvancedNarrativeConsistencyRAG
Main pipeline:
1. Index narratives (with caching)
2. Extract backstory claims
3. Retrieve supporting + opposing chunks
4. LLM judges with full evidence

## Configuration

Edit `rag_advanced.py` to customize:

### Chunking Parameters (line ~95)
```python
DependencyChunker(max_chunk_size=200, min_edge_density=0.3)
```
- **max_chunk_size**: Words per chunk (lower = more chunks, higher = fewer)
- **min_edge_density**: Syntactic density threshold

### Context Vector (line ~160)
```python
base_embedding[:900]  # Use 900 dims of original embedding
# + 32 dims sentiment
# + 32 dims temporal
# + 32 dims causal
# = 1024 total
```

Adjust weight by changing slice sizes.

### Graph-RAG (line ~340)
```python
similarities[i][j] > 0.65  # Similarity threshold for edges
```
Lower = denser graph, higher = sparser graph

### Retrieval K (line ~420)
```python
supporting_chunks: List[Tuple[str, float]], k: int = 5
```
Top-k supporting/opposing chunks to retrieve per claim

### LLM Model (line ~70)
```python
self.chat_model = "meta/llama-3.1-8b-instruct"
# Change to: meta/llama-3.1-70b-instruct for better quality
```

## Performance Tuning

### Faster Execution
```python
# Reduce chunk size for fewer chunks
max_chunk_size=150  # Instead of 200

# Reduce retrieval k
retrieve_supporting_and_opposing(..., k=3)  # Instead of 5

# Use smaller LLM
self.chat_model = "nvidia/mistral-7b-instruct"
```

### Better Quality
```python
# Increase chunk size for more context
max_chunk_size=300

# Higher k for more evidence
retrieve_supporting_and_opposing(..., k=7)

# Better LLM
self.chat_model = "meta/llama-3.1-70b-instruct"
```

## Understanding the Output

### results_advanced.csv
```
id,prediction,confidence,rationale
46,1,0.92,"Supporting chunks show Thalcave's economic background throughout narrative. No contradicting evidence found. Emotional arc consistent with described fears."
137,0,0.78,"Backstory claims character feared 'abandonment' but multiple narrative chunks show proactive engagement with relationships. Semantic opposition detected."
```

**Prediction:**
- 1 = Consistent (backstory aligns with narrative)
- 0 = Contradicts (backstory conflicts with narrative)

**Confidence:**
- 0.0-1.0 (higher = more certain)
- Based on strength of supporting/opposing evidence

**Rationale:**
- Multi-sentence explanation
- References both supporting evidence + identified contradictions

## Troubleshooting

### "ModuleNotFoundError: spacy"
```bash
pip install spacy
python -m spacy download en_core_web_md
```

### "ModuleNotFoundError: nltk"
```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

### "NVIDIA_API_KEY not found"
```bash
# Create .env
echo "NVIDIA_API_KEY=nvapi-your-key" > .env
echo "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1" >> .env
```

### "spaCy model not found: en_core_web_md"
```bash
python -m spacy download en_core_web_md
# If fails, try smaller model:
python -m spacy download en_core_web_sm
```

### Memory Issues
```python
# Reduce batch embedding size in client.embed()
# Add: batch_size = 16 instead of 32
```

### Slow Execution
- First run builds index (can take 5-10 min for large books)
- Subsequent runs use cached index (fast)
- Check: `narrative_index.pkl` should exist after first run

## Advanced Customization

### Add Custom Context Signals

Edit `ContextVectorBuilder.build_context_vector()` (~180):

```python
# Add domain-specific signals
custom_vec = np.array([your_score] * 32)
augmented = np.concatenate([
    base_embedding[:900],
    sentiment_vec,
    temporal_vec,
    causal_vec,
    custom_vec  # Add here
])
```

### Custom Negation Strategy

Edit `SemanticNegationFinder.negate_concept()` (~340):

```python
# Or use rule-based negation instead of LLM
def negate_concept(self, text: str) -> str:
    # Your custom negation logic
    return negated_text
```

### Integration with Pathway

To enable Pathway streaming (optional):

```python
import pathway as pw

# Create Pathway table from CSV
input_stream = pw.io.csv.read(self.csv_path)

# Process rows as they arrive
processed = input_stream.select(
    id=pw.this.id,
    analysis=pw.apply(self.analyze_backstory, ...)
)
```

## Expected Performance

| Metric | Value |
|--------|-------|
| Index Build | 5-15 min (depends on narrative size) |
| Per-Row Processing | 3-5 sec (depends on book size) |
| Memory | 2-4 GB (depends on book count) |
| Embeddings Stored | 1024-dim (NVIDIA) |
| Context Vector Size | 1024-dim (augmented) |

## Next Steps

1. Run `python rag_advanced.py`
2. Check `results_advanced.csv`
3. Analyze: prediction distribution, confidence scores, reasoning quality
4. Fine-tune parameters based on results
5. Consider ensembling multiple runs for better calibration

---

**Production-Ready Advanced RAG System** ✓
