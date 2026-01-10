"""
Main pipeline orchestrator for Advanced Narrative Consistency RAG
"""

import logging
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

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
        
        REFINED STRATEGY:
        1. Chunk backstory using dependency parsing
        2. For each backstory chunk:
           - Find supporting narrative chunks (high cosine similarity)
           - Find opposing narrative chunks (geometric negation)
           - Build rich context (temporal, emotional, causal)
        3. Aggregate evidence
        4. LLM reasoning with rich context and proper prompts
        
        Args:
            book_key: Book identifier
            character: Character name
            backstory: Backstory dictionary
            corpus: Full corpus of chunks
            
        Returns:
            ConsistencyAnalysis with prediction and reasoning
        """
        logger.info(f"Analyzing: {book_key} × {character}")

        # Extract and chunk backstory
        backstory_text = self._extract_backstory_text(backstory)
        backstory_chunks = self.chunker.chunk_text(backstory_text)
        logger.info(f"  Backstory chunked into {len(backstory_chunks)} segments")

        # Get narrative chunks
        narrative_chunks = corpus.get(book_key, [])
        if not narrative_chunks:
            logger.warning(f"No narrative chunks found for book: {book_key}")
            return ConsistencyAnalysis(
                backstory_id=character,
                prediction=0,
                confidence=0.1,
                supporting_chunks=[],
                opposing_chunks=[],
                reasoning="No narrative data found",
                graph_path=[]
            )

        # Collect supporting and opposing evidence
        all_supporting = []
        all_opposing = []

        for bs_chunk_text, bs_graph, bs_ents in backstory_chunks:
            # Embed backstory chunk
            bs_embedding = self.client.embed([bs_chunk_text])[0]
            
            # Find supporting chunks (high similarity)
            supp_chunks = self._find_supporting_chunks(
                bs_embedding, narrative_chunks, k=6
            )
            all_supporting.extend(supp_chunks)

            # Find opposing chunks (negation + semantic distance)
            opp_chunks = self._find_opposing_chunks(
                bs_chunk_text, narrative_chunks, bs_embedding, k=4
            )
            all_opposing.extend(opp_chunks)

        # Deduplicate
        all_supporting = self._deduplicate_chunks(all_supporting)[:5]
        all_opposing = self._deduplicate_chunks(all_opposing)[:5]

        logger.info(f"  Supporting chunks: {len(all_supporting)} | Opposing chunks: {len(all_opposing)}")

        # LLM reasoning with rich context
        pred, conf, reason = self.consistency_analyzer.reason_consistency_enhanced(
            book_key, character, backstory_text, 
            all_supporting, all_opposing, 
            narrative_chunks, self.context_builder
        )

        return ConsistencyAnalysis(
            backstory_id=character,
            prediction=pred,
            confidence=conf,
            supporting_chunks=[(c.chunk_id, s) for c, s in all_supporting],
            opposing_chunks=[(c.chunk_id, s) for c, s in all_opposing],
            reasoning=reason,
            graph_path=[]
        )

    def _extract_backstory_text(self, backstory) -> str:
        """Combine all backstory fields into single text"""
        if isinstance(backstory, str):
            return backstory
        if isinstance(backstory, dict):
            parts = []
            for key in ["early_events", "beliefs", "motivations", "fears", "assumptions_about_world"]:
                if key in backstory:
                    items = backstory[key]
                    if isinstance(items, list):
                        parts.extend(str(i) for i in items if i)
                    else:
                        parts.append(str(items))
            return " ".join(parts)
        return str(backstory)

    def _find_supporting_chunks(self, query_embedding: np.ndarray, 
                               chunks: List, k: int = 6) -> List[Tuple]:
        """Find top-k supporting chunks by cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = np.array([c.embedding for c in chunks])
        sims = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(sims)[-k:][::-1]
        return [(chunks[int(i)], float(sims[i])) for i in top_indices if sims[i] > 0.1]

    def _find_opposing_chunks(self, bs_text: str, chunks: List, 
                             bs_embedding: np.ndarray, k: int = 4) -> List[Tuple]:
        """Find opposing chunks via negation + distance"""
        try:
            # Get negation of backstory
            negated = self.negation_finder.negate_concept(bs_text)
            neg_embedding = self.client.embed([negated])[0]
            
            from sklearn.metrics.pairwise import cosine_similarity
            embeddings = np.array([c.embedding for c in chunks])
            sims = cosine_similarity([neg_embedding], embeddings)[0]
            top_indices = np.argsort(sims)[-k:][::-1]
            return [(chunks[int(i)], float(sims[i])) for i in top_indices if sims[i] > 0.1]
        except Exception as e:
            logger.warning(f"Failed to find opposing chunks: {e}")
            return []

    def _deduplicate_chunks(self, pairs: List[Tuple]) -> List[Tuple]:
        """Deduplicate by chunk_id"""
        seen = set()
        out = []
        for chunk, score in pairs:
            cid = getattr(chunk, "chunk_id", None)
            if cid and cid not in seen:
                seen.add(cid)
                out.append((chunk, score))
        return out


def main():
    """Main entry point"""
    rag = AdvancedNarrativeConsistencyRAG()
    rag.run_pipeline()


if __name__ == "__main__":
    main()
