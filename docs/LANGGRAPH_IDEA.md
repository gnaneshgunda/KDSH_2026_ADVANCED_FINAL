# LangGraph RAG - Core Idea

## Concept

Transform monolithic RAG pipeline into **node-based workflow** using LangGraph, where each processing step is an independent node connected by edges.

## Why LangGraph?

**Before**: Sequential method calls in classes
```python
claims = extract_claims(backstory)
evidence = retrieve_evidence(claims)
result = analyze(evidence)
```

**After**: Graph-based workflow
```python
workflow.add_node("extract", extract_claims_node)
workflow.add_node("retrieve", retrieve_evidence_node)
workflow.add_node("analyze", analyze_node)
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", "analyze")
```

## Key Innovation

### State Management
Shared `GraphState` dictionary flows through nodes:
```python
class GraphState(TypedDict):
    backstory_text: str
    claims: List
    evidence: List
    prediction: int
    confidence: float
```

### Node Pattern
Each node is a pure function:
```python
def extract_claims_node(state: GraphState) -> GraphState:
    state['claims'] = parse_claims(state['backstory_text'])
    return state
```

### Conditional Routing
Dynamic workflow based on state:
```python
workflow.add_conditional_edges(
    "analyze",
    lambda s: "error_handler" if s.get('error') else END
)
```

## Architecture

```
CSV Input
    ↓
[load_corpus] → Load book chunks
    ↓
[extract_claims] → Parse backstory with spaCy
    ↓
[embed_claims] → NVIDIA embeddings
    ↓
[retrieve_supporting] → Find evidence (cosine similarity)
    ↓
[retrieve_opposing] → Find contradictions
    ↓
[analyze_consistency] → LLM reasoning
    ↓
[error_handler] → Handle failures
    ↓
CSV Output
```

## Benefits

1. **Modularity** - Each node is independent and testable
2. **Extensibility** - Add nodes without modifying existing code
3. **Visibility** - Inspect state at each step
4. **Flexibility** - Change workflow by modifying edges
5. **Error Handling** - Conditional routing for failures

## Model Upgrade

### Embeddings
- Old: `nvidia/nv-embed-qa` (1024d)
- New: `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048d)

### Chat
- Old: `meta/llama-3.1-8b-instruct`
- New: `moonshotai/kimi-k2-instruct-0905` (with reasoning)

### Client
- Old: Custom HTTP wrapper
- New: LangChain `NVIDIAEmbeddings` + `ChatNVIDIA`

## Implementation

6 core files implement the system:
- `config_langgraph.py` - Model configs
- `models_langgraph.py` - GraphState definition
- `langchain_client.py` - NVIDIA API wrapper
- `langgraph_nodes.py` - 7 processing nodes
- `index_manager_langgraph.py` - Corpus management
- `pipeline_langgraph.py` - Graph orchestrator

## Result

Modern, maintainable RAG system with clear separation of concerns and easy extensibility.
