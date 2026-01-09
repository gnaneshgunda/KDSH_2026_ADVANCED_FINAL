# LangGraph RAG - Usage Guide

## Quick Start

```bash
# 1. Install
pip install -r requirements_langgraph.txt
python -m spacy download en_core_web_md

# 2. Configure
echo "NVIDIA_API_KEY=nvapi-xxxxx" > .env

# 3. Run
python pipeline_langgraph.py
```

## Basic Usage

```python
from pipeline_langgraph import LangGraphRAGPipeline

# Run with defaults
pipeline = LangGraphRAGPipeline()
pipeline.run_pipeline()
```

## Custom Configuration

```python
# Custom paths
pipeline = LangGraphRAGPipeline(
    books_dir="./db/books",
    csv_path="./db/train.csv",
    output_file="results.csv"
)
pipeline.run_pipeline()
```

## Model Configuration

Edit `config_langgraph.py`:

```python
EMBEDDING_CONFIG = {
    "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "api_key": NVIDIA_API_KEY,
    "truncate": "NONE"
}

CHAT_CONFIG = {
    "model": "moonshotai/kimi-k2-instruct-0905",
    "api_key": NVIDIA_API_KEY,
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 4096
}
```

## Testing

```bash
# Run test suite
python test_langgraph.py

# Expected output:
# ✓ Client tests passed!
# ✓ Node tests passed!
# ✓ Pipeline test passed!
```

## Extending with Custom Nodes

```python
from models_langgraph import GraphState

def custom_processing_node(state: GraphState) -> GraphState:
    """Add custom processing logic"""
    # Your logic here
    state['custom_field'] = process_data(state['claims'])
    return state

# Add to workflow
from langgraph.graph import StateGraph
workflow = StateGraph(GraphState)
workflow.add_node("custom", custom_processing_node)
workflow.add_edge("extract_claims", "custom")
workflow.add_edge("custom", "embed_claims")
```

## Accessing the Graph

```python
pipeline = LangGraphRAGPipeline()
graph = pipeline.graph

# Execute with custom state
initial_state = {
    "record_id": "test_1",
    "book_key": "test_book",
    "character": "Test Character",
    "backstory_text": "Character backstory...",
    "corpus": corpus,
    # ... other fields
}

result = graph.invoke(initial_state)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## Data Format

### Input CSV
```csv
id,book_name,char,caption,content,label
46,In Search of the Castaways,Thalcave,,Backstory text...,consistent
```

### Output CSV
```csv
id,prediction,confidence,rationale
46,1,0.85,The backstory aligns with narrative evidence...
```

## Troubleshooting

### API Key Error
```bash
echo "NVIDIA_API_KEY=your-key-here" > .env
```

### Missing spaCy Model
```bash
python -m spacy download en_core_web_md
```

### Import Error
```bash
pip install --upgrade langchain-nvidia-ai-endpoints langgraph
```

## Performance Tuning

```python
# In config_langgraph.py
DEFAULT_TOP_K = 5              # Retrieval count
SIMILARITY_THRESHOLD = 0.65    # Support threshold
NEGATION_THRESHOLD = 0.15      # Opposition threshold
MAX_SUPPORTING_CHUNKS = 5      # Evidence limit
MAX_OPPOSING_CHUNKS = 5        # Contradiction limit
```

## Visualization

```bash
python visualize_graph.py
```

## Expected Output

```
INFO | LangGraph Configuration loaded
INFO | Building corpus index...
INFO | Loaded 2 books from cache
INFO | [1/50] Processing: 46 - Thalcave
INFO | [Node] Loading corpus
INFO | [Node] Extracting claims
INFO | [Node] Embedding claims
INFO | [Node] Retrieving supporting chunks
INFO | [Node] Retrieving opposing chunks
INFO | [Node] Analyzing consistency
INFO | Prediction: 1 | Confidence: 0.85
...
INFO | Results written to results_langgraph.csv
```
