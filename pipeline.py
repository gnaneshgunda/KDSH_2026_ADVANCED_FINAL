"""
Main pipeline orchestrator for Advanced Narrative Consistency RAG
"""

import logging
import csv
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, DEFAULT_BOOKS_DIR, DEFAULT_CSV_PATH,
    DEFAULT_INDEX_PATH, DEFAULT_OUTPUT_FILE, DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_EDGE_DENSITY, EMBEDDING_DIM
)
from nvidia_client import NVIDIAClient
from chunker import DependencyChunker
from context_builder import ContextVectorBuilder
from negation_finder import SemanticNegationFinder
from index_manager import IndexManager
from rag_analyzer import BackstoryExtractor, ConsistencyAnalyzer
from models import ConsistencyAnalysis

logger = logging.getLogger(__name__)


class AdvancedNarrativeConsistencyRAG:
    """
    Production-grade RAG system for analyzing narrative consistency with backstories
    
    Features:
    - Intelligent dependency-based chunking
    - Context vectors (temporal + emotional + causal)
    - Semantic negation detection
    - Graph-RAG multi-hop reasoning
    - LLM-based consistency reasoning
    """

    def __init__(self, books_dir: Path = DEFAULT_BOOKS_DIR, csv_path: Path = DEFAULT_CSV_PATH,
                 index_path: Path = DEFAULT_INDEX_PATH, output_file: str = DEFAULT_OUTPUT_FILE):
        """
        Initialize RAG system
        
        Args:
            books_dir: Directory containing narrative texts
            csv_path: Path to CSV with backstories
            index_path: Path to cache index
            output_file: Output file for results
        """
        self.books_dir = Path(books_dir)
        self.csv_path = Path(csv_path)
        self.index_path = Path(index_path)
        self.output_file = output_file

        # Initialize components
        self.client = NVIDIAClient(NVIDIA_API_KEY, NVIDIA_BASE_URL)
        self.chunker = DependencyChunker(max_chunk_size=DEFAULT_CHUNK_SIZE)
        self.context_builder = ContextVectorBuilder(embedding_dim=EMBEDDING_DIM)
        self.negation_finder = SemanticNegationFinder(self.client)
        
        # Index management
        self.index_manager = IndexManager(
            self.chunker, self.context_builder, self.client,
            self.books_dir, self.index_path
        )

        # Analysis components
        self.backstory_extractor = BackstoryExtractor(self.client, self.context_builder)
        self.consistency_analyzer = ConsistencyAnalyzer(self.client)

        logger.info("AdvancedNarrativeConsistencyRAG initialized")

    def run_pipeline(self):
        """Execute full RAG pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Advanced Narrative Consistency RAG Pipeline")
        logger.info("=" * 80)

        # Build or load index
        self.index_manager.load_or_build()
        corpus = self.index_manager.get_corpus()

        if not corpus:
            logger.error("No corpus loaded. Ensure books directory contains .txt files")
            return

        # Load CSV data
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return

        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")

        # Process each record
        self._process_records(df, corpus)

    def _process_records(self, df: pd.DataFrame, corpus: Dict):
        """Process CSV records and generate predictions"""
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction", "confidence", "rationale"])

            for idx, row in df.iterrows():
                try:
                    self._process_record(row, corpus, writer, idx, len(df))
                    f.flush()
                except Exception as e:
                    logger.error(f"Record {row.get('id', idx)} failed: {e}")
                    writer.writerow([row.get('id', idx), 0, "0.0", f"Error: {str(e)}"])
                    f.flush()

        logger.info(f"Results written to {self.output_file}")

    def _process_record(self, row: pd.Series, corpus: Dict, writer, idx: int, total: int):
        """Process a single record"""
        # Extract book and character
        parts = str(row.get("book_name", "")).split("×")
        book_key = parts[0].strip().lower()
        char_name = parts[1].strip() if len(parts) > 1 else "Character"

        record_id = row.get("id", idx)
        logger.info(f"[{idx + 1}/{total}] {record_id}: {char_name}")

        # Check if book exists in corpus
        if book_key not in corpus:
            logger.warning(f"Book '{book_key}' not in corpus")
            writer.writerow([record_id, 0, "0.0", f"Book not found: {book_key}"])
            return

        # Construct backstory
        backstory = {
            "early_events": [str(row.get("content", ""))],
            "beliefs": [],
            "motivations": [str(row.get("caption", ""))],
            "fears": [],
            "assumptions_about_world": []
        }

        # Analyze consistency
        analysis = self.analyze_backstory(book_key, char_name, backstory, corpus)

        # Write results
        writer.writerow([
            record_id,
            analysis.prediction,
            f"{analysis.confidence:.2f}",
            analysis.reasoning[:300]
        ])

        time.sleep(0.2)  # Rate limiting

    def analyze_backstory(self, book_key: str, character: str,
                         backstory: Dict, corpus: Dict) -> ConsistencyAnalysis:
        """
        Full analysis pipeline for a backstory
        
        Args:
            book_key: Book identifier
            character: Character name
            backstory: Backstory dictionary
            corpus: Full corpus of chunks
            
        Returns:
            ConsistencyAnalysis with prediction and reasoning
        """
        logger.info(f"Analyzing: {book_key} × {character}")

        # Extract claims
        claims = self.backstory_extractor.extract_claims(backstory)
        if not claims:
            logger.warning("No claims extracted")
            return ConsistencyAnalysis(
                backstory_id=character,
                prediction=0,
                confidence=0.1,
                supporting_chunks=[],
                opposing_chunks=[],
                reasoning="No backstory claims found",
                graph_path=[]
            )

        # Retrieve supporting & opposing chunks
        all_supporting = []
        all_opposing = []
        
        chunks = corpus.get(book_key, [])
        for claim in claims:
            supp, opp = self.consistency_analyzer.retrieve_supporting_and_opposing(
                chunks, claim, self.negation_finder, k=3
            )
            all_supporting.extend(supp)
            all_opposing.extend(opp)

        # Deduplicate
        all_supporting = list(dict.fromkeys(all_supporting))[:5]
        all_opposing = list(dict.fromkeys(all_opposing))[:5]

        # LLM reasoning
        pred, conf, reason = self.consistency_analyzer.reason_consistency(
            book_key, character, claims, all_supporting, all_opposing, corpus
        )

        return ConsistencyAnalysis(
            backstory_id=character,
            prediction=pred,
            confidence=conf,
            supporting_chunks=all_supporting,
            opposing_chunks=all_opposing,
            reasoning=reason,
            graph_path=[]
        )


def main():
    """Main entry point"""
    rag = AdvancedNarrativeConsistencyRAG()
    rag.run_pipeline()


if __name__ == "__main__":
    main()
