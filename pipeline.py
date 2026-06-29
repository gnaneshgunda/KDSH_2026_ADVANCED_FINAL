"""
Main pipeline orchestrator for Advanced Narrative Consistency RAG
Refactored for per-book pkl caching and formatted CSV output
Input: train.csv (id, book_name, char, caption, content, label)
Output: results.csv (id, verdict, confidence, rationale)

KEY FIX: Verdict logic now correctly counts CONTRADICTIONS
- NOT_MENTIONED ≠ CONTRADICTED (only contradictions matter)
- Deterministic logic based on actual claim verdicts
- Confidence calculated from contradiction count
"""


import logging
import csv
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, DEFAULT_BOOKS_DIR, DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH, DEFAULT_OUTPUT_FILE, DEFAULT_CHUNK_SIZE,
    EMBEDDING_DIM, DEFAULT_TOP_K, MAX_TOP_K, RETRIEVAL_STEP, MAX_RETRIEVAL_ROUNDS,
    FALLBACK_ENABLED, LLM_TEMPERATURE, USE_PATHWAY
)
from nvidia_client import NVIDIAClient
from chunker import SemanticChunker
from context_builder import ContextVectorBuilder
from index_manager import IndexManager
from models import ConsistencyAnalysis, ChunkMetadata
from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier
from retriever import HybridRetriever # Using the new hybrid retriever


logger = logging.getLogger(__name__)



class Retriever:
    """Retrieve relevant narrative chunks for a query/claim"""
    
    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])
        logger.info(f"Retriever initialized with {len(chunks)} chunks")


    def retrieve(self, query: str, top_k: int = 5) -> List[ChunkMetadata]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Query string (claim)
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant ChunkMetadata objects
        """
        if not self.chunks:
            return []
            
        try:
            query_embedding = np.array(self.client.embed([query])[0])
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            retrieved = [self.chunks[int(i)] for i in top_indices if i < len(self.chunks)]
            logger.debug(f"Retrieved {len(retrieved)} chunks for query (top_k={top_k})")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []



class AdvancedNarrativeConsistencyRAG:
    """
    Production-grade RAG system for analyzing narrative consistency.
    
    Features:
    - Per-book pkl caching in ./db directory
    - Claim extraction from backstories
    - Claim verification with fallback retrieval loop
    - Deterministic verdict logic (only contradictions matter)
    - CSV output with structured format (id, verdict, confidence, rationale)
    """


    def __init__(self, 
                 books_dir: Path = DEFAULT_BOOKS_DIR, 
                 csv_path: Path = DEFAULT_CSV_PATH,
                 db_path: Path = DEFAULT_DB_PATH,
                 output_file: str = DEFAULT_OUTPUT_FILE):
        """
        Initialize RAG system.
        
        Args:
            books_dir: Directory containing narrative .txt files
            csv_path: CSV file with backstories (train.csv format)
            db_path: Directory to store per-book pkl files
            output_file: Output CSV file for results
        """
        self.books_dir = Path(books_dir)
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path)
        self.output_file = output_file


        # Initialize components
        self.client = NVIDIAClient(NVIDIA_API_KEY, NVIDIA_BASE_URL)
        self.chunker = SemanticChunker(max_chunk_size=DEFAULT_CHUNK_SIZE)
        self.context_builder = ContextVectorBuilder(embedding_dim=EMBEDDING_DIM)
        
        # Index management with per-book pkl
        self.index_manager = IndexManager(
            self.chunker, self.context_builder, self.client,
            self.books_dir, self.db_path
        )
        
        # Pathway real-time indexing (optional)
        self.pathway_indexer = None
        if USE_PATHWAY:
            try:
                from pathway_indexer import enable_pathway_mode
                self.pathway_indexer = enable_pathway_mode(
                    self.books_dir, self.chunker, self.context_builder, self.client
                )
                logger.info("Pathway real-time indexing enabled")
            except ImportError:
                logger.warning("Pathway not installed. Install with: pip install pathway")


        # Claim-based components
        self.claim_extractor = ClaimExtractor(self.client)
        self.claim_verifier = ClaimVerifier(self.client)


        logger.info("AdvancedNarrativeConsistencyRAG initialized")


    def run_pipeline(self):
        """Execute full RAG pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Advanced Narrative Consistency RAG Pipeline")
        logger.info("=" * 80)


        # Build or load indices (per-book pkl files)
        corpus = self.index_manager.load_or_build()
        logger.info(f"Corpus contains {len(corpus)} books: {list(corpus.keys())}")


        if not corpus:
            logger.error("No corpus loaded. Ensure books directory contains .txt files")
            return


        # Load CSV data
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return


        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records from CSV: {self.csv_path}")


        # Process each record
        self._process_records(df, corpus)


    def _process_records(self, df: pd.DataFrame, corpus: Dict[str, List[ChunkMetadata]]):
        """
        Process CSV records and write results to output file.
        
        Args:
            df: DataFrame with backstories
            corpus: Dictionary of book_name -> list of ChunkMetadata
        """
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Output columns: id, verdict, confidence, rationale
            writer.writerow(["id", "verdict", "confidence", "rationale"])


            for idx, row in df.iterrows():
                try:
                    self._process_record(row, corpus, writer, idx, len(df))
                    f.flush()
                except Exception as e:
                    logger.error(f"Record {row.get('id', idx)} failed: {e}")
                    record_id = row.get('id', idx)
                    writer.writerow([record_id, "unknown", "0.0", f"Error: {str(e)}"])
                    f.flush()


        logger.info(f"✓ Results written to {self.output_file}")


    def _process_record(self, row: pd.Series, corpus: Dict, 
                       writer: csv.writer, idx: int, total: int):
        """
        Process a single backstory record.
        
        Args:
            row: DataFrame row with backstory data
            corpus: Corpus of narrative chunks
            writer: CSV writer
            idx: Record index
            total: Total records
        """
        record_id = row.get("id", idx)
        book_name = str(row.get("book_name", "")).strip().lower()
        character = str(row.get("char", "Unknown")).strip()


        logger.info(f"[{idx + 1}/{total}] Processing: {record_id} | {book_name} × {character}")


        # Check if book exists in corpus
        if book_name not in corpus:
            logger.warning(f"Book '{book_name}' not in corpus. Available: {list(corpus.keys())}")
            writer.writerow([record_id, "unknown", "0.0", f"Book not found: {book_name}"])
            return


        # Construct backstory text from available columns
        backstory_text = self._extract_backstory_text(row)


        if not backstory_text.strip():
            logger.warning(f"Empty backstory for {record_id}")
            writer.writerow([record_id, "unknown", "0.5", "Backstory text empty"])
            return


        # Analyze consistency
        analysis = self.analyze_backstory(
            book_name, character, backstory_text, corpus[book_name]
        )


        # Write results: id, verdict, confidence, rationale
        writer.writerow([
            record_id,
            analysis.verdict,
            f"{analysis.confidence:.4f}",
            analysis.rationale[:1000]  # Truncate if needed
        ])


        time.sleep(0.05)  # Rate limiting


    def _extract_backstory_text(self, row: pd.Series) -> str:
        """
        Extract and combine backstory text from CSV columns.
        
        Args:
            row: DataFrame row
            
        Returns:
            Combined backstory text
        """
        parts = []
        
        # Try common column names
        for col in ["content", "caption", "backstory", "story", "narrative"]:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val:
                    parts.append(val)
        
        return " ".join(parts).strip()


    def analyze_backstory(self, book_name, character, backstory_text, narrative_chunks,
                          use_rerank=True, top_k=5, context_window=1,
                          enable_fallback=True, enable_negation=False):
        """
        Analyze backstory with detailed rationale generation, weighted voting, and trace logging.
        """
        start_time = time.time()
        
        # Extract claims (limited to 12)
        claims = self.claim_extractor.extract_claims(backstory_text)[:12]
        claim_categories = getattr(self.claim_extractor, 'last_claim_categories', {})
        
        retriever = HybridRetriever(narrative_chunks, self.client)
        
        results = []
        claim_results_for_ui = []
        per_claim_traces = []

        category_weights = {
            "IDENTITY": 2.0,
            "TEMPORAL": 1.5,
            "SPATIAL": 1.5,
            "RELATIONAL": 1.5,
            "CAUSAL": 1.0,
        }

        weighted_support = 0.0
        weighted_contradict = 0.0
        total_weight = 0.0

        for i, claim in enumerate(claims, 1):
            claim_start = time.time()
            category = claim_categories.get(claim, "IDENTITY")
            weight = category_weights.get(category, 1.0)
            total_weight += weight
            
            # Verify claim with or without fallback
            if enable_fallback:
                res, evidence = self._verify_claim_with_fallback(
                    claim, retriever, self.claim_verifier,
                    character_name=character, initial_top_k=top_k,
                    context_window=context_window, use_rerank=use_rerank
                )
            else:
                evidence = retriever.retrieve(
                    claim, 
                    character_name=character, 
                    top_k=top_k,
                    context_window=context_window,
                    use_rerank=use_rerank
                )
                res = self.claim_verifier.verify(claim, [c.text for c in evidence])

            # Optional Negation Check (Checking if evidence directly contradicts)
            if enable_negation and res.get("verdict") != "CONTRADICTED":
                negation_query = f"Evidence that {character} did not {claim}"
                neg_evidence = retriever.retrieve(
                    negation_query, 
                    character_name=character, 
                    top_k=3,
                    context_window=context_window,
                    use_rerank=use_rerank
                )
                neg_res = self.claim_verifier.verify(claim, [c.text for c in neg_evidence])
                if neg_res.get("verdict") == "CONTRADICTED":
                    res = neg_res
                    evidence = neg_evidence

            verdict = res.get("verdict", "NOT_MENTIONED")
            confidence = res.get("confidence", 0.5)
            
            if verdict == "SUPPORTED":
                weighted_support += weight * confidence
            elif verdict == "CONTRADICTED":
                weighted_contradict += weight * confidence

            claim_time = (time.time() - claim_start) * 1000

            claim_results_for_ui.append({
                "claim": claim,
                "category": category,
                "verdict": verdict,
                "rationale": res.get("rationale", ""),
                "confidence": confidence,
                "evidence": [c.text for c in evidence]
            })
            
            per_claim_traces.append({
                "claim": claim,
                "category": category,
                "chunks_retrieved": len(evidence),
                "verification_verdict": verdict,
                "verification_confidence": confidence,
                "time_ms": claim_time
            })
            
            results.append(res)

        # Calculate stats for the final analysis
        num_supported = sum(1 for r in claim_results_for_ui if r["verdict"] == "SUPPORTED")
        num_contradicted = sum(1 for r in claim_results_for_ui if r["verdict"] == "CONTRADICTED")
        num_unknown = sum(1 for r in claim_results_for_ui if r["verdict"] in ["NOT_MENTIONED", "UNKNOWN"])

        # Weighted voting verdict
        CONTRADICTION_THRESHOLD = 1.2
        if weighted_contradict >= CONTRADICTION_THRESHOLD:
            final_verdict = "0"  # Inconsistent
            final_confidence = min(0.98, 0.7 + (weighted_contradict / max(total_weight, 1.0)) * 0.3)
            rationale = self._build_contradiction_rationale(claim_results_for_ui)
        elif weighted_support > 0:
            final_verdict = "1"  # Consistent
            final_confidence = min(0.95, 0.6 + (weighted_support / max(total_weight, 1.0)) * 0.4)
            rationale = self._build_support_rationale(claim_results_for_ui, len(claims))
        else:
            final_verdict = "unknown"
            final_confidence = 0.5
            rationale = "No claims could be explicitly verified against the narrative. The text is silent on these points."
            
        total_time_ms = (time.time() - start_time) * 1000
        
        trace = {
            "claims_extracted": claims,
            "per_claim_traces": per_claim_traces,
            "total_time_ms": total_time_ms,
            "final_verdict": final_verdict,
            "final_confidence": final_confidence
        }

        return ConsistencyAnalysis(
            backstory_id=character,
            verdict=final_verdict,
            confidence=final_confidence,
            rationale=rationale,
            claim_results=claim_results_for_ui,
            num_supported_claims=num_supported,
            num_contradicted_claims=num_contradicted,
            num_unknown_claims=num_unknown,
            trace=trace
        )
    
    def _build_contradiction_rationale(self, claim_results):
        """Build detailed rationale for contradictions."""
        contradicted = [c for c in claim_results if c["verdict"] == "CONTRADICTED"]
        rationale_parts = [
            f"INCONSISTENT: Found {len(contradicted)} contradictory claims against narrative evidence."
        ]
        
        for i, c in enumerate(contradicted[:2], 1):
            rationale_parts.append(f"\nConflict {i}: {c['claim'][:80]}...")
            rationale_parts.append(f"Reasoning: {c['rationale']}")
            
        if len(contradicted) > 2:
            rationale_parts.append(f"\n(+{len(contradicted) - 2} more contradictions)")
            
        return " ".join(rationale_parts)
    
    def _build_support_rationale(self, claim_results, total_claims):
        """Build detailed rationale for consistent backstories."""
        supported = [c for c in claim_results if c["verdict"] == "SUPPORTED"]
        rationale_parts = [
            f"CONSISTENT: Verified {len(supported)}/{total_claims} claims explicitly in the source narrative."
        ]
        
        for i, c in enumerate(supported[:3], 1):
            rationale_parts.append(f"\n{i}. {c['claim'][:60]}... -> {c['rationale'][:80]}...")
            
        if len(supported) > 3:
            rationale_parts.append(f"\n(+{len(supported) - 3} more verified claims)")
            
        return " ".join(rationale_parts)

    def _verify_claim_with_fallback(self, claim: str, retriever,
                                    verifier, character_name: str = None,
                                    initial_top_k: int = 5, context_window: int = 1,
                                    use_rerank: bool = True) -> Tuple[Dict, List]:
        """
        Verify a claim with fallback retrieval loop.
        
        If verifier returns UNKNOWN, increase retrieved chunks and retry.
        Max rounds: MAX_RETRIEVAL_ROUNDS
        Max top_k: MAX_TOP_K
        """
        top_k = initial_top_k
        rounds = 0
        max_rounds = MAX_RETRIEVAL_ROUNDS
        
        chunks = []
        result = {"verdict": "NOT_MENTIONED", "rationale": "No evidence retrieved", "confidence": 0.5}

        while rounds < max_rounds and top_k <= MAX_TOP_K:
            logger.debug(f"    Retrieval round {rounds + 1}: top_k={top_k}")
            
            # Retrieve evidence
            chunks = retriever.retrieve(
                claim, 
                character_name=character_name, 
                top_k=top_k,
                context_window=context_window,
                use_rerank=use_rerank
            )
            evidence = [c.text for c in chunks]

            if not evidence:
                break

            # Verify claim
            result = verifier.verify(claim, evidence)

            # Check if verdict is conclusive
            if result.get("verdict", "").upper() in ["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]:
                logger.debug(f"    Final verdict: {result['verdict']}")
                return result, chunks

            # Not conclusive, increase top_k and retry
            logger.debug(f"    Verdict inconclusive, expanding retrieval...")
            top_k += RETRIEVAL_STEP
            rounds += 1

        return result, chunks


def main():
    """Main entry point"""
    rag = AdvancedNarrativeConsistencyRAG()
    rag.run_pipeline()



if __name__ == "__main__":
    main()