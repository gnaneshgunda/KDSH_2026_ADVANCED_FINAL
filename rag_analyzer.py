"""
Core RAG analysis module for consistency reasoning
"""

import logging
import json
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import BackstoryClaim, ConsistencyAnalysis, ChunkMetadata
from config import nlp, DEFAULT_TOP_K, MAX_SUPPORTING_CHUNKS, MAX_OPPOSING_CHUNKS

logger = logging.getLogger(__name__)


class BackstoryExtractor:
    """Extract structured claims from backstory JSON"""

    def __init__(self, client, context_builder):
        """
        Args:
            client: NVIDIAClient instance
            context_builder: ContextVectorBuilder instance
        """
        self.client = client
        self.context_builder = context_builder
        
        self.claim_types = {
            "early_events": "event",
            "beliefs": "belief",
            "motivations": "motivation",
            "fears": "fear",
            "assumptions_about_world": "trait"
        }
        
        logger.info("BackstoryExtractor initialized")

    def extract_claims(self, backstory_json: Dict) -> List[BackstoryClaim]:
        """
        Extract structured claims from backstory
        
        Args:
            backstory_json: Dictionary with backstory information
            
        Returns:
            List of BackstoryClaim objects
        """
        claims = []
        claim_id = 0

        for claim_type_key, claim_category in self.claim_types.items():
            if claim_type_key not in backstory_json:
                continue

            items = backstory_json[claim_type_key]
            if isinstance(items, str):
                items = [items]

            for item in items:
                claim_text = str(item).strip()
                if not claim_text:
                    continue

                try:
                    embedding = self.client.embed([claim_text])[0]
                    context_vec = self.context_builder.build_context_vector(claim_text, embedding)

                    doc = nlp(claim_text)
                    entities = [ent.text for ent in doc.ents]

                    claim = BackstoryClaim(
                        claim_id=f"claim_{claim_id}",
                        text=claim_text,
                        embedding=embedding,
                        context_vector=context_vec,
                        claim_type=claim_category,
                        entities=entities,
                        importance=0.8
                    )
                    claims.append(claim)
                    claim_id += 1
                except Exception as e:
                    logger.warning(f"Failed to process claim '{claim_text[:50]}...': {e}")

        logger.info(f"Extracted {len(claims)} backstory claims")
        return claims


class ConsistencyAnalyzer:
    """Analyze narrative consistency with backstories"""

    def __init__(self, client):
        """
        Args:
            client: NVIDIAClient instance
        """
        self.client = client
        logger.info("ConsistencyAnalyzer initialized")

    def retrieve_supporting_and_opposing(self, chunks: List[ChunkMetadata],
                                        backstory_claim: BackstoryClaim,
                                        negation_finder,
                                        k: int = DEFAULT_TOP_K) -> Tuple[List[Tuple[str, float]], 
                                                                          List[Tuple[str, float]]]:
        """
        Retrieve both supporting and opposing narrative chunks
        
        Args:
            chunks: List of narrative chunks
            backstory_claim: Backstory claim to analyze
            negation_finder: SemanticNegationFinder instance
            k: Number of top results to retrieve
            
        Returns:
            Tuple of (supporting_chunks, opposing_chunks)
        """
        if not chunks:
            return [], []

        # Supporting chunks (high cosine similarity)
        embeddings = np.array([c.embedding for c in chunks])
        similarities = cosine_similarity([backstory_claim.embedding], embeddings)[0]

        top_indices = np.argsort(similarities)[-k:][::-1]
        supporting = [
            (chunks[int(i)].chunk_id, float(similarities[i]))
            for i in top_indices
        ]

        # Opposing chunks (geometrical opposites)
        try:
            opposing = negation_finder.find_negated_chunks(
                backstory_claim.text,
                [c.text for c in chunks],
                embeddings
            )
            # Convert indices to chunk IDs
            opposing = [
                (chunks[int(idx)].chunk_id, float(sim))
                for idx, sim in opposing
            ]
        except Exception as e:
            logger.warning(f"Failed to find opposing chunks: {e}")
            opposing = []

        logger.debug(f"Found {len(supporting)} supporting and {len(opposing)} opposing chunks")
        return supporting, opposing

    def reason_consistency(self, book_key: str, character: str,
                         claims: List[BackstoryClaim],
                         supporting_chunks: List[Tuple[str, float]],
                         opposing_chunks: List[Tuple[str, float]],
                         corpus: Dict) -> Tuple[int, float, str]:
        """
        Use LLM to reason about consistency with rich evidence
        
        Args:
            book_key: Identifier for book/narrative
            character: Character name
            claims: Extracted backstory claims
            supporting_chunks: Supporting narrative evidence
            opposing_chunks: Contradicting narrative evidence
            corpus: Full corpus of chunks
            
        Returns:
            Tuple of (prediction, confidence, reasoning)
        """
        # Build evidence text
        supporting_text = self._format_evidence(
            supporting_chunks, corpus, book_key, "SUPPORTING"
        ) if supporting_chunks else "None found"

        opposing_text = self._format_evidence(
            opposing_chunks, corpus, book_key, "OPPOSING"
        ) if opposing_chunks else "None found"

        backstory_summary = "\n".join([
            f"- {c.claim_type}: {c.text}"
            for c in claims[:5]
        ])

        prompt = f"""You are a literary consistency expert analyzing narrative coherence.

BOOK: {book_key}
CHARACTER: {character}

BACKSTORY CLAIMS:
{backstory_summary}

SUPPORTING NARRATIVE CHUNKS:
{supporting_text}

OPPOSING/CONTRADICTING NARRATIVE CHUNKS:
{opposing_text}

TASK: Determine if the backstory is consistent with the narrative.
Consider:
1. Do supporting chunks substantiate the backstory?
2. Do opposing chunks create logical contradictions?
3. Is there a coherent causal chain?
4. Are emotional/temporal arcs aligned?

Return JSON:
{{
  "consistent": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation"
}}"""

        try:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = json.loads(response)
            
            prediction = 1 if result.get("consistent", False) else 0
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            logger.info(f"Consistency verdict: {['CONTRADICTION', 'CONSISTENT'][prediction]} "
                       f"(confidence: {confidence:.2f})")
            return prediction, confidence, reasoning
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return 0, 0.3, f"Error parsing response: {str(e)}"
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return 0, 0.3, f"Error: {str(e)}"

    def _format_evidence(self, chunks: List[Tuple[str, float]], corpus: Dict,
                        book_key: str, label: str) -> str:
        """Format evidence for prompt"""
        formatted = []
        for chunk_id, sim in chunks[:3]:
            try:
                idx = int(chunk_id.split("_")[-1])
                if book_key in corpus and idx < len(corpus[book_key]):
                    chunk_text = corpus[book_key][idx].text[:200]
                    formatted.append(f"{label} ({sim:.2f}): {chunk_text}")
            except (ValueError, IndexError):
                pass
        return "\n".join(formatted)

    def _chunk_id_to_index(self, chunk_id: str) -> int:
        """Convert chunk_id to index"""
        try:
            return int(chunk_id.split("_")[-1])
        except (ValueError, IndexError):
            return 0
