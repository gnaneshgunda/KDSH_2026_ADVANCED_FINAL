"""
LangGraph-based Advanced Narrative Consistency RAG Pipeline
"""

import logging
import csv
import time
from pathlib import Path
from typing import Dict

import pandas as pd
from langgraph.graph import StateGraph, END

from config import (
    EMBEDDING_CONFIG, CHAT_CONFIG, DB_BOOKS_DIR, DB_CSV_PATH,
    DEFAULT_INDEX_PATH, DEFAULT_OUTPUT_FILE, RECURSION_LIMIT
)
from models import GraphState
from client import LangChainNVIDIAClient
from nodes import (
    load_corpus_node,
    extract_claims_node,
    embed_claims_node,
    retrieve_supporting_node,
    retrieve_opposing_node,
    analyze_consistency_node,
    error_handler_node
)
from index_manager import LangGraphIndexManager

logger = logging.getLogger(__name__)


class LangGraphRAGPipeline:
    """
    LangGraph-based RAG system for narrative consistency analysis
    
    Architecture:
    - Nodes: Discrete processing steps (load, extract, embed, retrieve, analyze)
    - Edges: Workflow connections between nodes
    - State: Shared state passed through the graph
    """
    
    def __init__(
        self,
        books_dir: Path = DB_BOOKS_DIR,
        csv_path: Path = DB_CSV_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
        output_file: str = DEFAULT_OUTPUT_FILE
    ):
        """Initialize LangGraph RAG pipeline"""
        self.books_dir = Path(books_dir)
        self.csv_path = Path(csv_path)
        self.index_path = Path(index_path)
        self.output_file = output_file
        
        # Initialize LangChain NVIDIA client
        self.client = LangChainNVIDIAClient(EMBEDDING_CONFIG, CHAT_CONFIG)
        
        # Initialize index manager
        self.index_manager = LangGraphIndexManager(
            self.client, self.books_dir, self.index_path
        )
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("LangGraph RAG Pipeline initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow
        
        Graph structure:
        START -> load_corpus -> extract_claims -> embed_claims
              -> retrieve_supporting -> retrieve_opposing
              -> analyze_consistency -> END
        """
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("load_corpus", load_corpus_node)
        workflow.add_node("extract_claims", extract_claims_node)
        workflow.add_node("embed_claims", embed_claims_node)
        workflow.add_node("retrieve_supporting", retrieve_supporting_node)
        workflow.add_node("retrieve_opposing", retrieve_opposing_node)
        workflow.add_node("analyze_consistency", analyze_consistency_node)
        workflow.add_node("error_handler", error_handler_node)
        
        # Define edges (workflow)
        workflow.set_entry_point("load_corpus")
        workflow.add_edge("load_corpus", "extract_claims")
        workflow.add_edge("extract_claims", "embed_claims")
        workflow.add_edge("embed_claims", "retrieve_supporting")
        workflow.add_edge("retrieve_supporting", "retrieve_opposing")
        workflow.add_edge("retrieve_opposing", "analyze_consistency")
        
        # Conditional edge: check for errors
        workflow.add_conditional_edges(
            "analyze_consistency",
            lambda state: "error_handler" if state.get('error') else END,
            {
                "error_handler": "error_handler",
                END: END
            }
        )
        workflow.add_edge("error_handler", END)
        
        # Compile graph
        app = workflow.compile()
        logger.info("LangGraph workflow compiled")
        return app
    
    def run_pipeline(self):
        """Execute full RAG pipeline"""
        logger.info("=" * 80)
        logger.info("Starting LangGraph-based Narrative Consistency RAG")
        logger.info("=" * 80)
        
        # Build or load corpus index
        self.index_manager.load_or_build()
        corpus = self.index_manager.get_corpus()
        
        if not corpus:
            logger.error("No corpus loaded. Ensure db/books/ contains .txt files")
            return
        
        # Load CSV data
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Process records
        self._process_records(df, corpus)
    
    def _process_records(self, df: pd.DataFrame, corpus: Dict):
        """Process CSV records through LangGraph"""
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction", "confidence", "rationale"])
            
            for idx, row in df.iterrows():
                try:
                    result = self._process_record(row, corpus, idx, len(df))
                    writer.writerow([
                        result['record_id'],
                        result['prediction'],
                        f"{result['confidence']:.2f}",
                        result['reasoning'][:300]
                    ])
                    f.flush()
                except Exception as e:
                    logger.error(f"Record {row.get('id', idx)} failed: {e}")
                    writer.writerow([row.get('id', idx), 0, "0.0", f"Error: {str(e)}"])
                    f.flush()
                
                time.sleep(0.2)  # Rate limiting
        
        logger.info(f"Results written to {self.output_file}")
    
    def _process_record(self, row: pd.Series, corpus: Dict, idx: int, total: int) -> Dict:
        """Process single record through LangGraph"""
        # Parse book and character
        book_name = str(row.get("book_name", ""))
        parts = book_name.split("×") if "×" in book_name else [book_name, "Character"]
        book_key = parts[0].strip().lower()
        char_name = parts[1].strip() if len(parts) > 1 else row.get("char", "Character")
        
        record_id = row.get("id", idx)
        logger.info(f"[{idx + 1}/{total}] Processing: {record_id} - {char_name}")
        
        # Build backstory text
        content = str(row.get("content", ""))
        caption = str(row.get("caption", ""))
        backstory_text = f"{caption}\n{content}".strip()
        
        # Initialize graph state
        initial_state: GraphState = {
            "record_id": str(record_id),
            "book_key": book_key,
            "character": char_name,
            "backstory_text": backstory_text,
            "corpus": corpus,
            "chunks": [],
            "claims": [],
            "claim_embeddings": [],
            "supporting_chunks": [],
            "opposing_chunks": [],
            "prediction": None,
            "confidence": None,
            "reasoning": None,
            "error": None,
            "iteration": 0,
            "_client": self.client  # Inject client
        }
        
        # Execute graph
        try:
            final_state = self.graph.invoke(
                initial_state,
                {"recursion_limit": RECURSION_LIMIT}
            )
            
            return {
                "record_id": final_state['record_id'],
                "prediction": final_state.get('prediction', 0),
                "confidence": final_state.get('confidence', 0.0),
                "reasoning": final_state.get('reasoning', "No reasoning generated")
            }
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return {
                "record_id": str(record_id),
                "prediction": 0,
                "confidence": 0.0,
                "reasoning": f"Graph error: {str(e)}"
            }


def main():
    """Main entry point"""
    pipeline = LangGraphRAGPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
