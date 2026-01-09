# LangGraph RAG - Complete Description

## Overview

LangGraph-based Advanced Narrative Consistency RAG system that analyzes whether character backstories are consistent with their source narratives using node-based workflow architecture.

## System Architecture

### Workflow Graph
```
START
  ↓
load_corpus → Load book chunks from corpus
  ↓
extract_claims → Parse backstory into claims (spaCy)
  ↓
embed_claims → Generate embeddings (NVIDIA API)
  ↓
retrieve_supporting → Find supporting evidence (cosine similarity)
  ↓
retrieve_opposing → Find contradicting evidence (negation detection)
  ↓
analyze_consistency → LLM-based reasoning
  ↓
error_handler → Handle failures gracefully
  ↓
END
```

### Core Components

**config_langgraph.py** (~100 lines)
- NVIDIA endpoint configurations
- Model settings (embedding: llama-3.2-nv-embedqa-1b-v2, chat: kimi-k2-instruct-0905)
- System parameters (TOP_K, thresholds, dimensions)
- NLP model initialization (spaCy, NLTK)

**models_langgraph.py** (~80 lines)
- `GraphState`: TypedDict for state management with fields for input, processing, retrieval, output, and control
- `ChunkMetadata`: Rich chunk metadata with embeddings, entities, sentiment
- `BackstoryClaim`: Structured backstory claims
- `ConsistencyAnalysis`: Analysis results with prediction, confidence, reasoning

**langchain_client.py** (~90 lines)
- `LangChainNVIDIAClient`: Wrapper for LangChain NVIDIA endpoints
- `embed_texts()`: Batch embedding generation
- `embed_query()`: Single query embedding
- `chat_completion()`: Standard chat
- `chat_stream()`: Streaming with reasoning support

**langgraph_nodes.py** (~250 lines)
- `load_corpus_node`: Loads book chunks from corpus dictionary
- `extract_claims_node`: Parses backstory into sentences using spaCy
- `embed_claims_node`: Generates embeddings via NVIDIA API
- `retrieve_supporting_node`: Finds supporting evidence using cosine similarity (threshold: 0.65)
- `retrieve_opposing_node`: Finds contradicting evidence (threshold: 0.15)
- `analyze_consistency_node`: LLM-based consistency reasoning with prompt building
- `error_handler_node`: Handles errors with default values

**index_manager_langgraph.py** (~180 lines)
- `LangGraphIndexManager`: Corpus building and caching
- `load_or_build()`: Loads cached index or builds new
- `_build_corpus()`: Processes book files into chunks
- `_chunk_text()`: Dependency-aware chunking with spaCy
- Batch embedding generation (20 per batch)
- Entity extraction, sentiment analysis, temporal/causal markers
- Pickle-based caching for fast loading

**pipeline_langgraph.py** (~200 lines)
- `LangGraphRAGPipeline`: Main orchestrator
- `_build_graph()`: Constructs StateGraph workflow with nodes and edges
- `run_pipeline()`: Executes full pipeline (load corpus, process CSV, generate results)
- `_process_records()`: Processes CSV records through graph
- `_process_record()`: Processes single record with state initialization and graph invocation

## GraphState Structure

```python
class GraphState(TypedDict):
    # Input
    record_id: str                    # Record identifier
    book_key: str                     # Book reference
    character: str                    # Character name
    backstory_text: str               # Backstory content
    
    # Corpus
    corpus: Dict                      # All book chunks
    
    # Processing
    chunks: List                      # Current book chunks
    claims: List                      # Extracted claims
    claim_embeddings: List[np.ndarray] # Claim vectors
    
    # Retrieval
    supporting_chunks: List           # Supporting evidence
    opposing_chunks: List             # Contradicting evidence
    
    # Analysis
    prediction: Optional[int]         # 0 (contradict) or 1 (consistent)
    confidence: Optional[float]       # 0.0 to 1.0
    reasoning: Optional[str]          # LLM explanation
    
    # Control
    error: Optional[str]              # Error message
    iteration: int                    # Iteration counter
```

## Data Flow

1. **Input**: CSV with columns (id, book_name, char, caption, content, label)
2. **Corpus Loading**: Book texts from db/books/ chunked and embedded
3. **State Initialization**: Create GraphState for each record
4. **Graph Execution**: State flows through nodes sequentially
5. **Output**: CSV with columns (id, prediction, confidence, rationale)

## Node Processing Details

### load_corpus_node
- Checks if book_key exists in corpus
- Loads chunks for specified book
- Sets error if book not found

### extract_claims_node
- Parses backstory_text with spaCy
- Extracts sentences as claims
- Limits to 10 claims maximum

### embed_claims_node
- Calls NVIDIA embedding API
- Generates 2048-dimensional vectors
- Handles batch processing

### retrieve_supporting_node
- Computes cosine similarity between claim embeddings and chunk embeddings
- Filters by SIMILARITY_THRESHOLD (0.65)
- Returns top-k chunks (default: 5)
- Deduplicates results

### retrieve_opposing_node
- Computes cosine similarity
- Identifies low similarity (< NEGATION_THRESHOLD: 0.15)
- Returns contradicting chunks
- Deduplicates results

### analyze_consistency_node
- Builds LLM prompt with claims, supporting evidence, opposing evidence
- Calls NVIDIA chat API (kimi-k2-instruct-0905)
- Parses response for prediction, confidence, reasoning
- Handles errors gracefully

### error_handler_node
- Checks for errors in state
- Sets default values (prediction: 0, confidence: 0.0)
- Logs error details

## Model Specifications

### Embedding Model
- Name: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Dimensions: 2048
- Truncate: NONE
- Use: Document and query embeddings

### Chat Model
- Name: `moonshotai/kimi-k2-instruct-0905`
- Temperature: 0.6
- Top-p: 0.9
- Max tokens: 4096
- Features: Reasoning support

## Configuration Parameters

```python
EMBEDDING_DIM = 2048
DEFAULT_CHUNK_SIZE = 200
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.65
NEGATION_THRESHOLD = 0.15
MULTI_HOP_DEPTH = 2
MAX_SUPPORTING_CHUNKS = 5
MAX_OPPOSING_CHUNKS = 5
RECURSION_LIMIT = 25
```

## Performance Characteristics

- **Startup Time**: ~3s (graph compilation + model loading)
- **Per-Record Processing**: ~1.8s (node execution + API calls)
- **Memory Usage**: ~550MB (corpus + models)
- **Cache Size**: ~100MB (pickle serialization)
- **Batch Size**: 20 texts per embedding API call
- **Rate Limiting**: 0.2s delay between records

## Error Handling

- **Conditional Routing**: Errors trigger error_handler_node
- **Graceful Degradation**: Default values on failure
- **Detailed Logging**: Structured logs at each node
- **State Preservation**: Error stored in state for inspection

## Extension Points

### Add Custom Node
```python
def custom_node(state: GraphState) -> GraphState:
    # Custom processing
    return state

workflow.add_node("custom", custom_node)
workflow.add_edge("previous_node", "custom")
```

### Modify Workflow
```python
# Change edge connections
workflow.add_edge("extract_claims", "custom_node")
workflow.add_edge("custom_node", "embed_claims")
```

### Add Conditional Routing
```python
workflow.add_conditional_edges(
    "node_name",
    lambda s: "route_a" if condition(s) else "route_b",
    {"route_a": "node_a", "route_b": "node_b"}
)
```

## Testing

**test_langgraph.py** provides:
- Client connectivity tests (embedding + chat)
- Individual node execution tests
- Pipeline initialization tests
- State management validation

## Dependencies

```
langchain>=0.1.0
langgraph>=0.0.20
langchain-nvidia-ai-endpoints>=0.1.0
nltk>=3.8.0
spacy>=3.5.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
```

## Comparison with Original

| Aspect | Original | LangGraph |
|--------|----------|-----------|
| Architecture | Monolithic classes | Node-based workflow |
| State | Class attributes | TypedDict GraphState |
| Execution | Sequential methods | Graph traversal |
| Client | Custom HTTP | LangChain wrappers |
| Embedding | 1024d | 2048d |
| Visualization | None | Built-in |
| Extensibility | Subclassing | Add nodes/edges |

## Use Cases

- Analyze narrative consistency in literature
- Validate character backstories in game development
- Quality assurance for story content
- Research in computational narrative analysis
- Educational tools for literature study

## Resources

- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain NVIDIA: https://python.langchain.com/docs/integrations/providers/nvidia
- NVIDIA API: https://build.nvidia.com/
