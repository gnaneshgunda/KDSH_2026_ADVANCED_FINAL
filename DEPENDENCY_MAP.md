# Module Dependency Map

## Dependency Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                          rag_advanced.py                         │
│              (Backward Compatibility - Entry Point)              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         pipeline.py                              │
│            (Main Orchestrator - New Entry Point)                 │
│                                                                   │
│  AdvancedNarrativeConsistencyRAG                                 │
│  - Coordinates all components                                    │
│  - Manages full end-to-end pipeline                              │
│  - CSV input/output handling                                     │
└──┬──────────────────┬──────────────────┬──────────────────┬─────┘
   │                  │                  │                  │
   ▼                  ▼                  ▼                  ▼
┌──────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ config   │  │ index_manager    │  │ rag_analyzer     │  │   nvidia_    │
│          │  │                  │  │                  │  │   client     │
│ • Setup  │  │ • Build corpus   │  │ • Extract claims │  │              │
│ • Init   │  │ • Cache index    │  │ • Retrieve evid. │  │ • Embeddings │
│ • Const  │  │ • Load chunks    │  │ • Reason consist │  │ • Chat       │
└────┬─────┘  └──┬────────┬──────┘  └──┬──────────┬────┘  └──────────────┘
     │           │        │            │          │
     │           │        │            │          │
     ▼           ▼        ▼            ▼          ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ models   │  │ chunker  │  │ context_ │  │ negation │
   │          │  │          │  │ builder  │  │ _finder  │
   │ Data     │  │ Chunk    │  │          │  │          │
   │ Structures│  │ text     │  │ Sentiment│  │ Semantic │
   └──────────┘  └──┬───────┘  │ Temporal │  │ opposites│
        ▲           │          │ Causal   │  └────┬─────┘
        │           │          └──┬───────┘       │
        │           │             │              │
        │           │             │              │
        └───┬───────┴─────────────┘              │
            │                                    │
            │                  ┌─────────────────┘
            │                  │
            ▼                  ▼
        ┌────────────────────────────┐
        │      graph_rag             │
        │                            │
        │  • Build similarity graph  │
        │  • Multi-hop search        │
        │  • Reasoning paths         │
        └────────────────────────────┘
```

## File Dependency Tree (Imports)

```
rag_advanced.py
└── pipeline.py
    ├── config.py
    │   ├── nltk
    │   ├── spacy
    │   └── dotenv
    │
    ├── nvidia_client.py
    │   ├── config.py
    │   ├── requests
    │   └── numpy
    │
    ├── chunker.py
    │   ├── config.py (uses nlp, nltk)
    │   ├── models.py
    │   └── nltk.tokenize
    │
    ├── context_builder.py
    │   ├── config.py (uses nlp)
    │   └── numpy
    │
    ├── negation_finder.py
    │   ├── nvidia_client.py
    │   ├── numpy
    │   └── sklearn.metrics.pairwise
    │
    ├── graph_rag.py
    │   ├── models.py
    │   ├── config.py
    │   ├── numpy
    │   ├── networkx
    │   └── sklearn.metrics.pairwise
    │
    ├── index_manager.py
    │   ├── config.py
    │   ├── models.py
    │   ├── chunker.py
    │   ├── context_builder.py
    │   ├── graph_rag.py
    │   ├── pickle
    │   └── pathlib
    │
    └── rag_analyzer.py
        ├── config.py (uses nlp)
        ├── models.py
        ├── nvidia_client.py
        ├── negation_finder.py (optional)
        ├── numpy
        ├── json
        └── sklearn.metrics.pairwise
```

## Component Interaction Flow

```
                    ┌─────────────────┐
                    │  CSV Input      │
                    │  (book_name,    │
                    │   content)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Load Books     │
                    │  from ./books/  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ Chunker  │       │ Context  │       │   NLP    │
   │          │       │ Builder  │       │ (spaCy)  │
   │ → chunks │       │ → ctx_vec│       │ → entities
   └────┬─────┘       └──────────┘       └──────────┘
        │
        └──────────┬───────────┬──────────┬────────────┐
                   │           │          │            │
                   ▼           ▼          ▼            ▼
            ┌──────────┐  ┌──────┐  ┌─────────┐  ┌──────────┐
            │ Embedder │  │Graph │  │ Index   │  │ Metadata │
            │(NVIDIA)  │  │ RAG  │  │ Manager │  │ Builder  │
            └────┬─────┘  └──────┘  └────┬────┘  └──────────┘
                 │                        │
                 └────────────┬───────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Corpus Index │
                      │ (cached pickle)
                      └───────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         ┌────────┐  ┌──────────────┐  ┌────────────┐
         │Extract │  │Retrieve Evid.│  │ LLM Reason │
         │Claims  │  │(Supporting +  │  │(Consistency)
         │        │  │ Opposing)     │  │            │
         └────┬───┘  └───┬──────┬────┘  └──────┬─────┘
              │          │      │             │
              └──────────┴──────┴─────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Result    │
                  │ (prediction,│
                  │ confidence, │
                  │ reasoning)  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ CSV Output  │
                  │ results.csv │
                  └─────────────┘
```

## Data Flow

```
INPUT: CSV Row
  {id, book_name, content, caption}
         │
         ▼
PARSING: Extract book_key & character_name
         │
         ▼
CORPUS LOADING: Get chunks for this book
  [ChunkMetadata, ChunkMetadata, ...]
         │
         ▼
CLAIM EXTRACTION: Parse content/caption
  [BackstoryClaim, BackstoryClaim, ...]
         │
         ▼
RETRIEVAL: Find supporting/opposing chunks
  ┌─────────────────────────────┐
  │ For each claim:             │
  │ • Get embedding             │
  │ • Find similar chunks       │
  │ • Find contradicting chunks │
  └──────────┬──────────────────┘
             │
             ▼
REASONING: Prompt LLM with evidence
  {
    "supporting_chunks": [...],
    "opposing_chunks": [...],
    "claims": [...]
  }
         │
         ▼
DECISION: LLM response → parsed result
  {
    "consistent": true/false,
    "confidence": 0.85,
    "reasoning": "..."
  }
         │
         ▼
OUTPUT: ConsistencyAnalysis
  {
    prediction: 1,
    confidence: 0.85,
    reasoning: "..."
  }
```

## Configuration Flow

```
.env (secrets)
    ├── NVIDIA_API_KEY
    └── NVIDIA_BASE_URL
         │
         ▼
config.py (constants & init)
    ├── API credentials
    ├── Model parameters
    ├── Thresholds
    ├── Paths
    └── NLP models (spaCy, NLTK)
         │
         ▼
All other modules
    └── Import config values
```

## Testing Strategy by Module

```
✓ config.py              → Environment & imports
✓ models.py              → Data structure integrity
✓ nvidia_client.py       → API connectivity
✓ chunker.py             → Text segmentation quality
✓ context_builder.py     → Vector dimensions & normalization
✓ negation_finder.py     → Contradiction detection
✓ graph_rag.py           → Graph connectivity & traversal
✓ index_manager.py       → Caching & serialization
✓ rag_analyzer.py        → JSON parsing & reasoning
✓ pipeline.py            → End-to-end integration
```

## Scalability Considerations

```
SINGLE MACHINE (Current)
    Core logic → File-based caching → Local index

DISTRIBUTED (Potential)
    Core logic → Redis cache → Separate embedding service
                           → Separate LLM service
                           → Database corpus

    chunker.py ──────┐
    context_builder  ├──→ Embedding Service (NVIDIA)
                     │
    models.py ──────┘
    
    
    rag_analyzer.py ──→ LLM Service (NVIDIA)
    
    
    pipeline.py ──────┐
    index_manager     ├──→ Corpus Database
                      │
                      └──→ Cache Layer (Redis)
```

---

## Quick Reference: What Each Module Does

| Module | Purpose | Key Class | Input | Output |
|--------|---------|-----------|-------|--------|
| config | Setup & constants | - | - | configuration |
| models | Data structures | ChunkMetadata, BackstoryClaim | - | dataclasses |
| nvidia_client | API wrapper | NVIDIAClient | texts/prompts | embeddings/responses |
| chunker | Text segmentation | DependencyChunker | text | chunks + graphs |
| context_builder | Contextual signals | ContextVectorBuilder | text + embedding | context_vector |
| negation_finder | Contradiction detection | SemanticNegationFinder | claim + chunks | opposing_indices |
| graph_rag | Multi-hop reasoning | GraphRAG | chunks | related_chunks/paths |
| index_manager | Corpus caching | IndexManager | books_dir | corpus + graph_rag |
| rag_analyzer | Analysis pipeline | BackstoryExtractor, ConsistencyAnalyzer | backstory | claims / prediction |
| pipeline | Orchestration | AdvancedNarrativeConsistencyRAG | csv_file | results_csv |

