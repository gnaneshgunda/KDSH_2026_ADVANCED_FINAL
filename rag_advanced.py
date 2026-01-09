"""
KDSH 2026 Track A – Advanced Narrative Consistency RAG
Production-Grade Implementation with:
  • Dependency Parsing for Intelligent Chunking
  • Context Vectors (Temporal + Emotional + Causal)
  • Semantic Negation (Geometrical Opposites)
  • Graph-RAG with Multi-hop Reasoning
  • Pathway Stream Processing
  • LangGraph for Complex Reasoning Workflows
  • NVIDIA NIM Backend (Zero OpenAI Dependency)

Author: Advanced Team
Date: January 2026

MODULAR ARCHITECTURE:
This file has been refactored into modular components for better maintainability:
  - config.py: Configuration and setup
  - models.py: Data models and structures
  - nvidia_client.py: NVIDIA NIM API client
  - chunker.py: Dependency-based text chunking
  - context_builder.py: Context vector construction
  - negation_finder.py: Semantic negation detection
  - graph_rag.py: Multi-hop reasoning graph
  - index_manager.py: Index building and caching
  - rag_analyzer.py: Backstory extraction and consistency analysis
  - pipeline.py: Main orchestration pipeline

See pipeline.py for the main execution entry point.
"""

import warnings
warnings.filterwarnings("ignore")

# For backward compatibility, import main components from new modules
from pipeline import AdvancedNarrativeConsistencyRAG, main

if __name__ == "__main__":
    main()
