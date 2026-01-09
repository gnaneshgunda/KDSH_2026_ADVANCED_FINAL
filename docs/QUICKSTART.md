# Quick Start Guide - Modular RAG

## File Organization

```
KDSH_2026_ADVANCED_FINAL/
├── config.py                    # ⚙️ Configuration & setup
├── models.py                    # 📦 Data structures
├── nvidia_client.py             # 🔌 API client
├── chunker.py                   # 📄 Text segmentation
├── context_builder.py           # 🧠 Context vectors
├── negation_finder.py           # 🔍 Contradiction detection
├── graph_rag.py                 # 🕸️ Multi-hop reasoning
├── index_manager.py             # 💾 Index caching
├── rag_analyzer.py              # 🔬 Core analysis
├── pipeline.py                  # 🚀 Main orchestration
├── rag_advanced.py              # ↩️ Backward compatibility
├── MODULAR_ARCHITECTURE.md      # 📚 Detailed docs
├── SETUP_ADVANCED.md            # 🔧 Installation guide
└── test_advanced.py             # ✅ Tests
```

## What Changed?

### Before (Monolithic)
```python
# Everything in one file
class NVIDIAClient: ...
class DependencyChunker: ...
class ContextVectorBuilder: ...
class GraphRAG: ...
class AdvancedNarrativeConsistencyRAG: ...
```

### After (Modular)
```python
# Separate modules by responsibility
nvidia_client.py    → NVIDIAClient
chunker.py          → DependencyChunker
context_builder.py  → ContextVectorBuilder
graph_rag.py        → GraphRAG
pipeline.py         → AdvancedNarrativeConsistencyRAG
```

## Running the Pipeline

```bash
# Same as before - backward compatible!
python rag_advanced.py

# Or directly use the pipeline module
python pipeline.py

# Or import and use programmatically
python -c "from pipeline import AdvancedNarrativeConsistencyRAG; rag = AdvancedNarrativeConsistencyRAG(); rag.run_pipeline()"
```

## Import Patterns

### Full Pipeline
```python
from pipeline import AdvancedNarrativeConsistencyRAG

rag = AdvancedNarrativeConsistencyRAG()
rag.run_pipeline()
```

### Individual Components
```python
# Text processing
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder

# APIs
from nvidia_client import NVIDIAClient

# Analysis
from rag_analyzer import BackstoryExtractor, ConsistencyAnalyzer
from negation_finder import SemanticNegationFinder
from graph_rag import GraphRAG

# Data models
from models import ChunkMetadata, BackstoryClaim, ConsistencyAnalysis

# Configuration
from config import EMBEDDING_DIM, DEFAULT_CHUNK_SIZE, nlp, NVIDIA_API_KEY
```

## Common Tasks

### Task 1: Process custom text
```python
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder
from nvidia_client import NVIDIAClient
from config import NVIDIA_API_KEY, NVIDIA_BASE_URL

# Setup
client = NVIDIAClient(NVIDIA_API_KEY, NVIDIA_BASE_URL)
chunker = DependencyChunker()
builder = ContextVectorBuilder()

# Process
text = "Your narrative text here..."
chunks = chunker.chunk_text(text)
embeddings = client.embed([c[0] for c in chunks])

for text, embedding in zip([c[0] for c in chunks], embeddings):
    context_vec = builder.build_context_vector(text, embedding)
    print(f"Chunk: {text[:50]}...")
    print(f"Context vector shape: {context_vec.shape}")
```

### Task 2: Analyze a backstory
```python
from pipeline import AdvancedNarrativeConsistencyRAG

rag = AdvancedNarrativeConsistencyRAG()
rag.index_manager.load_or_build()

backstory = {
    "early_events": ["Character grew up in poverty"],
    "beliefs": ["Hard work is essential"],
    "motivations": ["Escape poverty"],
    "fears": ["Returning to poverty"],
    "assumptions_about_world": []
}

result = rag.analyze_backstory("book_key", "CharacterName", backstory)
print(f"Consistent: {result.prediction == 1}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reasoning: {result.reasoning}")
```

### Task 3: Find contradictions
```python
from negation_finder import SemanticNegationFinder
from nvidia_client import NVIDIAClient
import numpy as np

client = NVIDIAClient(api_key, base_url)
finder = SemanticNegationFinder(client)

claim = "The character loves their family"
narrative_chunks = [
    "The character abandoned their family",
    "The character cares for their siblings",
    "The character never visits home"
]

embeddings = np.array(client.embed(narrative_chunks))
contradictions = finder.find_negated_chunks(claim, narrative_chunks, embeddings)

for idx, score in contradictions:
    print(f"Contradicting chunk {idx}: {narrative_chunks[idx]}")
    print(f"  Opposition score: {score:.3f}")
```

### Task 4: Multi-hop reasoning
```python
from graph_rag import GraphRAG
from pipeline import AdvancedNarrativeConsistencyRAG

rag = AdvancedNarrativeConsistencyRAG()
rag.index_manager.load_or_build()

corpus = rag.index_manager.get_corpus()
graph_rag = rag.index_manager.get_graph_rag()

# Find all chunks related to a concept within 2 hops
start_chunk_id = "book_chunk_5"
related = graph_rag["book"].multi_hop_search(
    query_embedding=None,
    start_chunk_id=start_chunk_id,
    hops=2
)

print(f"Found {len(related)} related chunks:")
for chunk_id in related:
    print(f"  - {chunk_id}")
```

## Configuration Quick Reference

Edit `config.py`:

```python
# Text chunking
DEFAULT_CHUNK_SIZE = 200              # Max words per chunk
DEFAULT_MIN_EDGE_DENSITY = 0.3        # Graph density threshold

# RAG retrieval
DEFAULT_TOP_K = 5                     # Top-K chunks to retrieve
SIMILARITY_THRESHOLD = 0.65           # Graph edge threshold
NEGATION_THRESHOLD = 0.15             # Contradiction detection threshold
MULTI_HOP_DEPTH = 2                   # Graph traversal depth

# Model dimensions
EMBEDDING_DIM = 1024                  # Embedding vector size
MAX_SUPPORTING_CHUNKS = 5             # Max supporting evidence
MAX_OPPOSING_CHUNKS = 5               # Max opposing evidence

# API endpoints
NVIDIA_API_KEY = "..."                # From .env
NVIDIA_BASE_URL = "https://..."       # From .env or default
EMBEDDING_MODEL = "nvidia/nv-embed-qa"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"
```

## Module Dependencies

```
For basic text processing:
  - config.py (for spaCy/NLTK setup)
  - chunker.py
  - context_builder.py
  - models.py

For API calls:
  - config.py
  - nvidia_client.py

For full pipeline:
  - All of the above, plus:
  - negation_finder.py
  - graph_rag.py
  - index_manager.py
  - rag_analyzer.py
  - pipeline.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'config'` | Ensure all `.py` files are in the same directory |
| `NVIDIA_API_KEY not found` | Set `NVIDIA_API_KEY` in `.env` or `config.py` |
| `spaCy model not found` | Run `python -m spacy download en_core_web_md` |
| `No books in corpus` | Ensure `.txt` files exist in `./books/` directory |
| `Import errors` | Run `pip install -r requirements_advanced.txt` |

## Performance Tips

1. **Reuse embeddings**: Cache embeddings with pickle (done automatically)
2. **Batch operations**: NVIDIA client handles batching
3. **Limit hops**: Use `multi_hop_search(hops=2)` to limit traversal
4. **Filter chunks**: Pre-filter corpus before analysis
5. **Rate limiting**: Adjust `time.sleep(0.2)` in pipeline for API limits

## Adding to Existing Projects

Just copy the module files you need:

```python
# Minimal: just text processing
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder

# Full RAG
from pipeline import AdvancedNarrativeConsistencyRAG
```

All modules are self-contained with clear dependencies!

---

**Questions?** See [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) for detailed docs.
