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
    FALLBACK_ENABLED, LLM_TEMPERATURE
)
from nvidia_client import NVIDIAClient
from chunker import SemanticChunker
from context_builder import ContextVectorBuilder
from index_manager import IndexManager
from models import ConsistencyAnalysis, ChunkMetadata
from claim_extractor import ClaimExtractor
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


    def analyze_backstory(self, book_name: str, character: str,
                         backstory_text: str, 
                         narrative_chunks: List[ChunkMetadata]) -> ConsistencyAnalysis:
        """
        Analyze backstory consistency through claim verification.
        
        Flow:
        1. Extract claims from backstory
        2. For each claim, verify with fallback retrieval loop
        3. Aggregate results
        4. Determine verdict based on contradictions
        
        Args:
            book_name: Book name
            character: Character name
            backstory_text: Backstory text
            narrative_chunks: All narrative chunks from the book
            
        Returns:
            ConsistencyAnalysis with verdict, confidence, rationale
        """
        logger.info(f"Analyzing: {book_name} × {character}")


        # Step 1: Extract claims
        claims = self.claim_extractor.extract_claims(backstory_text)
        logger.info(f"  Extracted {len(claims)} claims")


        if not claims:
            logger.warning(f"  No claims extracted for {character}")
            return ConsistencyAnalysis(
                backstory_id=character,
                verdict="unknown",
                confidence=0.5,
                rationale="No claims could be extracted from backstory",
                claim_results=[]
            )


        # Step 2: Verify each claim with fallback loop
        if not narrative_chunks:
            logger.warning(f"  No narrative chunks found for book: {book_name}")
            return ConsistencyAnalysis(
                backstory_id=character,
                verdict="unknown",
                confidence=0.5,
                rationale="No narrative chunks available for analysis",
                claim_results=[]
            )


        retriever = Retriever(narrative_chunks, self.client)
        claim_results = []


        for claim in claims:
            logger.debug(f"  Verifying claim: {claim[:60]}...")
            
            result = self._verify_claim_with_fallback(
                claim, retriever, self.claim_verifier
            )
            
            claim_results.append({
                "claim": claim,
                "verdict": result["verdict"],
                "explanation": result["explanation"],
                "confidence": result.get("confidence", 0.5)
            })


            logger.debug(f"    Verdict: {result['verdict']} (confidence: {result.get('confidence', 0.5):.2f})")


        # Step 3: Determine verdict based on contradictions
        verdict, confidence, rationale = self._determine_verdict(
            character, claim_results
        )


        return ConsistencyAnalysis(
            backstory_id=character,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            claim_results=claim_results,
            num_supported_claims=sum(1 for r in claim_results if r["verdict"].lower() == "supported"),
            num_contradicted_claims=sum(1 for r in claim_results if r["verdict"].lower() == "contradicted"),
            num_unknown_claims=sum(1 for r in claim_results if r["verdict"].lower() == "unknown")
        )


    def _verify_claim_with_fallback(self, claim: str, retriever: Retriever,
                                    verifier: ClaimVerifier) -> Dict:
        """
        Verify a claim with fallback retrieval loop.
        
        If verifier returns UNKNOWN, increase retrieved chunks and retry.
        Max rounds: MAX_RETRIEVAL_ROUNDS
        Max top_k: MAX_TOP_K
        
        Args:
            claim: Claim to verify
            retriever: Retriever instance
            verifier: ClaimVerifier instance
            
        Returns:
            Dict with verdict, explanation, confidence
        """
        if not FALLBACK_ENABLED:
            # Single retrieval without fallback
            chunks = retriever.retrieve(claim, top_k=DEFAULT_TOP_K)
            evidence = [c.text for c in chunks]
            return verifier.verify(claim, evidence)


        # Fallback loop enabled
        top_k = DEFAULT_TOP_K
        rounds = 0
        max_rounds = MAX_RETRIEVAL_ROUNDS


        while rounds < max_rounds and top_k <= MAX_TOP_K:
            logger.debug(f"    Retrieval round {rounds + 1}: top_k={top_k}")
            
            # Retrieve evidence
            chunks = retriever.retrieve(claim, top_k=top_k)
            evidence = [c.text for c in chunks]


            # Verify claim
            result = verifier.verify(claim, evidence)


            # Check if verdict is conclusive
            if result["verdict"].lower() in ["supported", "contradicted", "not_mentioned"]:
                logger.debug(f"    Final verdict: {result['verdict']}")
                return result


            # Not conclusive, increase top_k and retry
            logger.debug(f"    Verdict inconclusive, expanding retrieval...")
            top_k += RETRIEVAL_STEP
            rounds += 1


        # Exhausted retrieval rounds
        logger.debug(f"    Max retrieval rounds ({max_rounds}) reached")
        return {
            "verdict": "not_mentioned",
            "explanation": "Claim not conclusively addressed in narrative after exhaustive search",
            "confidence": 0.4
        }


    def _determine_verdict(self, character: str,claim_results: List[Dict]) -> Tuple[str, float, str]:
        """
         Determine overall verdict based on claim verification results.

    LOGIC PRINCIPLES:
    - CONTRADICTED claims prove inconsistency
    - NOT_MENTIONED claims are neutral but weaken confidence
    - Too many NOT_MENTIONED → insufficient evidence → UNKNOWN
        """

        if not claim_results:
            return "unknown", 0.5, "No claims to analyze"

    # Count verdict types
        supported_count = sum(1 for r in claim_results if r["verdict"].lower() == "supported")
        contradicted_count = sum(1 for r in claim_results if r["verdict"].lower() == "contradicted")
        not_mentioned_count = sum(1 for r in claim_results if r["verdict"].lower() == "not_mentioned")
        unknown_count = sum(1 for r in claim_results if r["verdict"].lower() == "unknown")

        total_claims = len(claim_results)
        not_mentioned_ratio = not_mentioned_count / total_claims

        logger.info(
        f"  Claim summary: {supported_count} supported, "
        f"{contradicted_count} contradicted, "
        f"{not_mentioned_count} not mentioned, "
        f"{unknown_count} unknown"
    )

    # ===== FINAL VERDICT LOGIC =====

    # 1️⃣ Any contradiction → contradicted
        if contradicted_count > 0:
            verdict = "contradicted"
            confidence = min(
            0.95,
            0.55 + (contradicted_count / total_claims) * 0.45
        )
            rationale = (
            f"Backstory contradicts narrative. "
            f"Found {contradicted_count} contradicting claim(s) out of {total_claims}. "
            f"({supported_count} supported, {not_mentioned_count} not mentioned)"
        )
            logger.info(f"  Overall verdict: CONTRADICTED (confidence: {confidence:.2f})")

    # 2️⃣ All claims supported → consistent
        elif supported_count == total_claims:
            verdict = "consistent"
            confidence = 0.95
            rationale = (
            f"Backstory is fully consistent with narrative. "
            f"All {supported_count} claims are supported."
        )
            logger.info(f"  Overall verdict: CONSISTENT (confidence: {confidence:.2f})")

    # 3️⃣ Too much missing evidence → unknown
        elif not_mentioned_ratio >= 0.6:
            verdict = "unknown"
            confidence = max(0.40, 0.55 - not_mentioned_ratio * 0.15)
            rationale = (
            f"Insufficient narrative evidence to verify backstory. "
            f"{supported_count} claims supported, "
            f"{not_mentioned_count} not mentioned, "
            f"{contradicted_count} contradicted."
        )
            logger.info(f"  Overall verdict: UNKNOWN (confidence: {confidence:.2f})")

    # 4️⃣ Mixed but no contradictions → consistent (low confidence)
        else:
            verdict = "consistent"
            confidence = max(
            0.55,
            0.80 - not_mentioned_ratio * 0.30
        )
            rationale = (
            f"Backstory appears consistent with narrative. "
            f"{supported_count} claims supported, "
            f"{not_mentioned_count} not mentioned "
            f"(no contradictions found)."
        )
            logger.info(f"  Overall verdict: CONSISTENT (confidence: {confidence:.2f})")

        return verdict, confidence, rationale



def main():
    """Main entry point"""
    rag = AdvancedNarrativeConsistencyRAG()
    rag.run_pipeline()



if __name__ == "__main__":
    main()