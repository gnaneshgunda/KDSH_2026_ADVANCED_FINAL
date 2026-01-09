# Advanced Narrative Consistency RAG - Modular Architecture

## Overview

The **Advanced Narrative Consistency RAG** has been refactored into a clean, modular architecture with clear separation of concerns. Each module handles a specific responsibility, making the codebase more maintainable, testable, and extensible.

## Module Structure

### Core Configuration
**[config.py](config.py)**
- Centralized configuration management
- Environment setup (NLTK, spaCy)
- API credentials and endpoints
- Model parameters and thresholds
- All magic numbers and constants in one place

```python
from config import NVIDIA_API_KEY, DEFAULT_CHUNK_SIZE, EMBEDDING_DIM
```

### Data Models
**[models.py](models.py)**
- `ChunkMetadata`: Narrative chunk with embeddings and contextual signals
- `BackstoryClaim`: Extracted backstory statement with type and importance
- `ConsistencyAnalysis`: Complete analysis result with predictions and reasoning

```python
from models import ChunkMetadata, BackstoryClaim, ConsistencyAnalysis
```

### API Client
**[nvidia_client.py](nvidia_client.py)**
- `NVIDIAClient`: Wrapper for NVIDIA NIM API
- `embed()`: Generate vector embeddings for texts
- `chat()`: Get LLM completions with context
- Handles authentication and error handling

```python
from nvidia_client import NVIDIAClient
client = NVIDIAClient(api_key, base_url)
embeddings = client.embed(texts)
response = client.chat(messages)
```

### Text Chunking
**[chunker.py](chunker.py)**
- `DependencyChunker`: Intelligent text segmentation
- Uses spaCy dependency parsing to preserve semantic boundaries
- Respects max chunk size while maintaining coherence
- Builds dependency graphs for each chunk

```python
from chunker import DependencyChunker
chunker = DependencyChunker(max_chunk_size=200)
chunks = chunker.chunk_text(text)
# Returns: List[(text, dependency_graph, entities)]
```

### Context Vectors
**[context_builder.py](context_builder.py)**
- `ContextVectorBuilder`: Augments embeddings with contextual signals
- Analyzes sentiment polarity
- Extracts temporal markers (past/present/future)
- Detects causal indicators
- Combines all signals into normalized context vectors

```python
from context_builder import ContextVectorBuilder
builder = ContextVectorBuilder()
context_vec = builder.build_context_vector(text, base_embedding)
```

### Semantic Negation
**[negation_finder.py](negation_finder.py)**
- `SemanticNegationFinder`: Finds contradictory narratives
- Uses LLM to generate semantic opposites
- Identifies narrative chunks that contradict backstories
- Geometrical opposite detection in embedding space

```python
from negation_finder import SemanticNegationFinder
finder = SemanticNegationFinder(client)
opposing = finder.find_negated_chunks(claim_text, chunks, embeddings)
```

### Graph-RAG
**[graph_rag.py](graph_rag.py)**
- `GraphRAG`: Multi-hop reasoning over narrative chunks
- Builds similarity graph connecting related chunks
- Performs breadth-first search for related passages
- Finds shortest reasoning paths between concepts

```python
from graph_rag import GraphRAG
graph_rag = GraphRAG(chunks)
related = graph_rag.multi_hop_search(query_embedding, start_id, hops=2)
path = graph_rag.find_reasoning_path(source_id, target_id)
```

### Index Management
**[index_manager.py](index_manager.py)**
- `IndexManager`: Builds and caches narrative indices
- Loads books from disk and processes them
- Generates embeddings and context vectors
- Serializes corpus for fast reuse via pickle
- Handles index versioning and updates

```python
from index_manager import IndexManager
manager = IndexManager(chunker, context_builder, client, books_dir)
manager.load_or_build()
corpus = manager.get_corpus()
```

### RAG Analysis
**[rag_analyzer.py](rag_analyzer.py)**
- `BackstoryExtractor`: Parses backstory JSON into structured claims
  - Extracts events, beliefs, motivations, fears, traits
  - Generates embeddings and context vectors per claim
  - Identifies entities and importance scores

- `ConsistencyAnalyzer`: Core reasoning about narrative consistency
  - Retrieves supporting and opposing chunks
  - Prompts LLM with rich evidence
  - Parses structured JSON responses
  - Generates confidence scores and reasoning

```python
from rag_analyzer import BackstoryExtractor, ConsistencyAnalyzer
extractor = BackstoryExtractor(client, context_builder)
claims = extractor.extract_claims(backstory)

analyzer = ConsistencyAnalyzer(client)
supporting, opposing = analyzer.retrieve_supporting_and_opposing(chunks, claim, finder)
pred, conf, reason = analyzer.reason_consistency(book_key, character, claims, ...)
```

### Pipeline Orchestration
**[pipeline.py](pipeline.py)**
- `AdvancedNarrativeConsistencyRAG`: Main orchestrator
- Coordinates all components
- Implements full end-to-end pipeline
- Processes CSV input and generates CSV output

```python
from pipeline import AdvancedNarrativeConsistencyRAG
rag = AdvancedNarrativeConsistencyRAG(books_dir="./books", csv_path="train.csv")
rag.run_pipeline()
```

### Backward Compatibility
**[rag_advanced.py](rag_advanced.py)**
- Thin wrapper that imports from `pipeline.py`
- Maintains original entry point for existing code
- Can be used as before: `python rag_advanced.py`

## Dependency Graph

```
config.py ──────┐
                 ├─> nvidia_client.py
models.py ──────┤
                 ├─> chunker.py ─────────┐
                 │                        ├─> index_manager.py
context_builder.py ────────────────────┤                       ├─> pipeline.py
                                        ├─> rag_analyzer.py ──┤
negation_finder.py ──┐                  │                       │
                     ├─> graph_rag.py ──┘                       └─> rag_advanced.py
                     │
                     └─> rag_analyzer.py
```

## Usage Examples

### Basic Pipeline
```python
from pipeline import AdvancedNarrativeConsistencyRAG

rag = AdvancedNarrativeConsistencyRAG(
    books_dir="./books",
    csv_path="train.csv",
    index_path="advanced_index.pkl",
    output_file="results_advanced.csv"
)
rag.run_pipeline()
```

### Using Individual Components
```python
from config import NVIDIA_API_KEY, NVIDIA_BASE_URL, nlp
from nvidia_client import NVIDIAClient
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder

# Initialize
client = NVIDIAClient(NVIDIA_API_KEY, NVIDIA_BASE_URL)
chunker = DependencyChunker(max_chunk_size=200)
builder = ContextVectorBuilder()

# Process text
text = "The quick brown fox jumps..."
chunks = chunker.chunk_text(text)

# Get embeddings
chunk_texts = [c[0] for c in chunks]
embeddings = client.embed(chunk_texts)

# Build context vectors
for text, embeddings in zip(chunk_texts, embeddings):
    context_vec = builder.build_context_vector(text, embedding)
```

### Custom Analysis
```python
from pipeline import AdvancedNarrativeConsistencyRAG

rag = AdvancedNarrativeConsistencyRAG()
rag.index_manager.load_or_build()

# Analyze specific backstory
backstory = {
    "early_events": ["The character lost their home in a fire"],
    "beliefs": ["Fire is dangerous"],
    "motivations": ["Prevent similar tragedies"],
    "fears": [],
    "assumptions_about_world": []
}

result = rag.analyze_backstory("harry_potter", "Harry", backstory)
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasoning: {result.reasoning}")
```

## Extension Points

### Adding New Context Signals
Edit [context_builder.py](context_builder.py):
```python
def extract_new_signal(self, text: str) -> float:
    # Your implementation
    return signal_score

def build_context_vector(self, ...):
    # Add to augmented vector
    new_signal_vec = np.array([signal] * 32)
    augmented = np.concatenate([..., new_signal_vec])
```

### Custom Chunking Strategy
Create `custom_chunker.py`:
```python
class SemanticChunker:
    def chunk_text(self, text: str):
        # Your implementation
        return chunks
```

### Alternative LLM Backends
Edit [nvidia_client.py](nvidia_client.py) or create `openai_client.py`:
```python
class OpenAIClient:
    def embed(self, texts):
        # Use OpenAI API
    
    def chat(self, messages):
        # Use OpenAI chat API
```

## Configuration

Edit [config.py](config.py) to customize:

```python
# Chunking parameters
DEFAULT_CHUNK_SIZE = 200
DEFAULT_MIN_EDGE_DENSITY = 0.3

# RAG thresholds
SIMILARITY_THRESHOLD = 0.65
NEGATION_THRESHOLD = 0.15
MULTI_HOP_DEPTH = 2

# Model parameters
EMBEDDING_DIM = 1024
DEFAULT_TOP_K = 5
```

## Testing

Test individual modules:

```python
# Test chunker
from chunker import DependencyChunker
chunker = DependencyChunker()
chunks = chunker.chunk_text("Your test text here.")
assert len(chunks) > 0

# Test context builder
from context_builder import ContextVectorBuilder
import numpy as np
builder = ContextVectorBuilder()
embedding = np.random.rand(1024)
context = builder.build_context_vector("Test text", embedding)
assert context.shape == (1024,)
```

## Performance Optimization

1. **Index Caching**: Pre-built indices are cached via pickle. Delete `advanced_index.pkl` to rebuild.
2. **Batch Embeddings**: NVIDIA client handles batch embedding requests efficiently.
3. **Graph Efficiency**: Use `multi_hop_search()` with controlled `hops` parameter.
4. **Memory**: Consider splitting large corpora across multiple books.

## Error Handling

Each module includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Common issues:
- **Missing spaCy model**: `python -m spacy download en_core_web_md`
- **Missing NVIDIA key**: Set `NVIDIA_API_KEY` in `.env`
- **Empty books directory**: Ensure `.txt` files in `./books/`

## Architecture Benefits

✅ **Modularity**: Each component is independent and testable  
✅ **Reusability**: Import and use individual modules in other projects  
✅ **Maintainability**: Clear responsibilities, easy to debug  
✅ **Extensibility**: Add new features without modifying existing code  
✅ **Scalability**: Can distribute components across services  
✅ **Documentation**: Self-documenting code with docstrings  

---

**Last Updated**: January 2026  
**Version**: 2.0 (Modular Architecture)
