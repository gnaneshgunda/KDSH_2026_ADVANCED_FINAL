"""
Semantic negation finding using geometrical opposites
"""

import logging
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import NEGATION_THRESHOLD

logger = logging.getLogger(__name__)


class SemanticNegationFinder:
    """Find semantic opposites and contradictions in narrative chunks"""

    def __init__(self, client):
        """
        Args:
            client: NVIDIAClient instance for embeddings and LLM
        """
        self.client = client
        logger.info("SemanticNegationFinder initialized")

    def negate_concept(self, text: str) -> str:
        """
        Generate semantic opposite of a concept using LLM
        
        Args:
            text: Original statement
            
        Returns:
            Semantic opposite/contradiction statement
        """
        prompt = f"""Given this statement: "{text}"
Generate its semantic opposite/contradiction (antonym at concept level, not just logical negation).
Return ONLY the opposite statement, nothing else."""

        messages = [{"role": "user", "content": prompt}]
        negated = self.client.chat(messages, temperature=0.3)
        logger.debug(f"Generated negation for: {text[:50]}...")
        return negated

    def find_negated_chunks(self, backstory_chunk: str, narrative_chunks: List[str],
                           embeddings: np.ndarray) -> List[Tuple[int, float]]:
        """
        Find narrative chunks that contradict the backstory
        
        Args:
            backstory_chunk: Backstory statement to contradict
            narrative_chunks: List of narrative chunk texts
            embeddings: Pre-computed embeddings for narrative chunks
            
        Returns:
            List of (chunk_index, similarity_score) tuples for contradicting chunks
        """
        # Get negation of backstory
        negated_backstory = self.negate_concept(backstory_chunk)
        neg_embedding = self.client.embed([negated_backstory])[0]

        # Find chunks similar to negation (i.e., opposing backstory)
        similarities = cosine_similarity([neg_embedding], embeddings)[0]

        # Return top contradicting chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        result = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > NEGATION_THRESHOLD
        ]
        
        logger.debug(f"Found {len(result)} negated chunks for backstory")
        return result
