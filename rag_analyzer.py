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

        # 1. Initial Retrieval (Fast Cosine)
        # Fetch Top-20 candidates (broad) to filter down later
        initial_k = 20
        embeddings = np.array([c.embedding for c in chunks])
        similarities = cosine_similarity([backstory_claim.embedding], embeddings)[0]

        top_indices = np.argsort(similarities)[-initial_k:][::-1]
        
        # Prepare for Reranking
        candidates = []
        for i in top_indices:
            candidates.append({"text": chunks[int(i)].text, "metadata": chunks[int(i)]})
            
        # 2. Reranking (Precision)
        # Use NVIDIA Rerank API to sort candidates by true relevance
        try:
            rankings = self.client.rerank(backstory_claim.text, candidates)
            # Take Top-K from reranker
            top_ranked = rankings[:k]
            
            supporting = []
            for r in top_ranked:
                idx = r["index"]
                chunk = candidates[idx]["metadata"]
                # Use rerank score if available
                score = r.get("score", 0.9)
                supporting.append((chunk, score))
                
        except Exception as e:
            logger.warning(f"Reranking skipped: {e}")
            # Fallback to cosine
            supporting = [
                (chunks[int(i)], float(similarities[i]))
                for i in top_indices[:k]
            ]

        # 3. Opposing Chunks (Negation Logic)
        # Note: We currently don't rerank negation candidates as they are syntactically generated
        # But we could apply the same logic if needed.
        try:
            opposing_list = negation_finder.find_negated_chunks(
                backstory_claim.text,
                [c.text for c in chunks],
                embeddings
            )
            # Convert to (ChunkMetadata, score)
            opposing = [
                (chunks[int(idx)], float(sim))
                for idx, sim in opposing_list
            ]
        except Exception as e:
            logger.warning(f"Failed to find opposing chunks: {e}")
            opposing = []

        logger.debug(f"Found {len(supporting)} supporting and {len(opposing)} opposing chunks")
        return supporting, opposing

    def reason_consistency(self, book_key: str, character: str,
                         claims,
                         supporting_chunks: List,
                         opposing_chunks: List,
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

    def _format_evidence(self, chunks: List[Tuple[ChunkMetadata, float]], corpus: Dict,
                        book_key: str, label: str) -> str:
        """Format evidence for prompt with temporal context"""
        formatted = []
        
        # Calculate book length for normalization if needed, mostly handled in index_manager
        # But we can also infer max length from the chunk metadata itself if we have access to all chunks
        # Here we just use the pre-calculated start_pos if available
        
        for chunk, sim in chunks[:3]:
            try:
                # Use metadata object directly
                chunk_text = chunk.text[:300]
                
                # Format temporal marker
                # Assuming index_manager sets end_pos accurately, we can estimate progress
                # However, since we don't have total_len here easily sans corpus lookups, 
                # we rely on what we stored. 
                # If we want a percentage, we need the book's total length. 
                # Let's approximate or just show line numbers if mapped.
                
                # Robustness check: look up total length from corpus if possible
                total_len = 100000 # Default fallback
                if book_key in corpus and corpus[book_key]:
                     # Estimate from last chunk
                     total_len = corpus[book_key][-1].end_pos
                
                progress = 0.0
                if total_len > 0:
                    progress = (chunk.start_pos / total_len) * 100
                    
                formatted.append(f"{label} (Score: {sim:.2f}, Progress: {progress:.1f}%): \"{chunk_text}...\"")
            except Exception as e:
                logger.warning(f"Error formatting chunk: {e}")
                pass
        return "\n".join(formatted)

    def _chunk_id_to_index(self, chunk_id: str) -> int:
        pass

    def reason_consistency_enhanced(self, book_key: str, character: str, backstory_text: str,
                                   supporting_chunks: List[Tuple],
                                   opposing_chunks: List[Tuple],
                                   narrative_chunks: List,
                                   context_builder) -> Tuple[int, float, str]:
        """
        Enhanced reasoning with rich context vectors and better prompting
        
        Args:
            book_key: Book identifier
            character: Character name
            backstory_text: Full backstory text
            supporting_chunks: List of (ChunkMetadata, score) tuples
            opposing_chunks: List of (ChunkMetadata, score) tuples
            narrative_chunks: All narrative chunks for context
            context_builder: ContextVectorBuilder for enrichment
            
        Returns:
            Tuple of (prediction, confidence, reasoning)
        """
        # Format evidence with context vectors
        supporting_text = self._format_enhanced_evidence(
            supporting_chunks, context_builder, "SUPPORTING"
        ) if supporting_chunks else "No supporting evidence found"

        opposing_text = self._format_enhanced_evidence(
            opposing_chunks, context_builder, "OPPOSING"
        ) if opposing_chunks else "No contradictions found"

        # Build comprehensive prompt
        prompt = f"""
You are a narrative consistency analyzer. Your task is to evaluate whether a character's backstory is supported, contradicted, or left ambiguous by the narrative evidence provided.

BACKSTORY ({character}):
{backstory_text[:500]}

NARRATIVE CHUNKS MOST RELATED TO THE BACKSTORY:
{supporting_text}

NARRATIVE CHUNKS POTENTIALLY IN TENSION WITH THE BACKSTORY:
{opposing_text}

INSTRUCTIONS:
- Evaluate the actual content of the narrative chunks, not their labels.
- Decide whether the backstory is:
  • Supported by the narrative,
  • Clearly contradicted by the narrative, or
  • Not clearly addressed by the narrative.
- Return "consistent": true if the backstory is supported OR not clearly contradicted.
- Return "consistent": false ONLY if there is a clear, direct contradiction.
- Lack of evidence or weak evidence alone should NOT be treated as a contradiction.
- Set "confidence":
  • 0.8–1.0 if evidence is clear and decisive,
  • 0.5–0.7 if evidence is partial or indirect,
  • 0.3–0.4 if evidence is weak, sparse, or ambiguous.
- Provide a brief reasoning grounded in the narrative content.

Return ONLY valid JSON:
{{"consistent": true/false, "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}
"""


        try:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = json.loads(response)
            
            prediction = 1 if result.get("consistent", False) else 0
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "Analysis complete")
            
            logger.info(f"Consistency verdict: {['CONTRADICTION', 'CONSISTENT'][prediction]} "
                       f"(confidence: {confidence:.2f})")
            return prediction, confidence, reasoning
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback heuristic
            supp_score = len(supporting_chunks)
            opp_score = len(opposing_chunks)
            consistent = supp_score > opp_score * 1.5
            confidence = 0.5 + min(0.3, abs(supp_score - opp_score) * 0.1)
            return (1 if consistent else 0), confidence, "Heuristic fallback analysis"

    def _format_enhanced_evidence(self, chunks: List[Tuple], 
                                 context_builder, label: str) -> str:
        """Format evidence with context vector insights"""
        formatted = []
        for i, (chunk, score) in enumerate(chunks[:4], 1):
            # Extract context signals
            sentiment = context_builder.analyze_sentiment(chunk.text)
            temporal = context_builder.extract_temporal_markers(chunk.text)
            causal = context_builder.extract_causal_indicators(chunk.text)
            
            context_str = f"[Sentiment: {sentiment:+.2f}, Temporal: {','.join(temporal) or 'none'}, Causal: {','.join(causal)[:30] or 'none'}]"
            
            snippet = chunk.text[:300].replace("\n", " ")
            formatted.append(f"{i}. (Score: {score:.2f}) {context_str}\n   \"{snippet}...\"")
        
        return "\n".join(formatted) if formatted else "None"
        """Convert chunk_id to index"""
        try:
            return int(chunk_id.split("_")[-1])
        except (ValueError, IndexError):
            return 0
