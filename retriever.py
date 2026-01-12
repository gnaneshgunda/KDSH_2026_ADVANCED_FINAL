import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from models import ChunkMetadata
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Multi-stage retrieval:
    1. Metadata filtering (character, temporal, location)
    2. Semantic ranking (cosine similarity)
    3. Reranking (NVIDIA rerank API)
    4. Context expansion (neighboring chunks)
    """

    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])
        logger.info(f"HybridRetriever initialized with {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        character_name: str = None,
        top_k: int = 7,
        context_window: int = 1,
        use_rerank: bool = True
    ) -> List[ChunkMetadata]:
        """
        Retrieve relevant chunks with metadata filtering and reranking.
        
        Args:
            query: Search query (claim text)
            character_name: Filter by character mentions
            top_k: Number of chunks to return
            context_window: Number of neighboring chunks to include
            use_rerank: Whether to use NVIDIA rerank API
        """
        if not self.chunks:
            return []

        # STEP 1: Metadata-based filtering
        filtered_chunks = self._filter_by_metadata(character_name)
        
        # If too few results, fall back to all chunks
        search_pool = filtered_chunks if len(filtered_chunks) >= top_k else self.chunks
        
        logger.debug(f"Search pool: {len(search_pool)} chunks (filtered: {len(filtered_chunks)}, total: {len(self.chunks)})")

        # STEP 2: Semantic similarity ranking
        pool_embeddings = np.array([c.embedding for c in search_pool])
        query_emb = np.array(self.client.embed([query])[0])
        sims = cosine_similarity([query_emb], pool_embeddings)[0]
        
        # Get top 2*top_k for reranking
        initial_k = min(top_k * 2, len(search_pool))
        ranked_indices = np.argsort(sims)[::-1][:initial_k]

        # STEP 3: Reranking (optional)
        if use_rerank and len(ranked_indices) > top_k:
            try:
                candidates = [{
                    "text": search_pool[i].text,
                    "index": i
                } for i in ranked_indices]
                
                reranked = self.client.rerank(query, candidates)
                ranked_indices = [r["index"] for r in reranked[:top_k]]
                logger.debug(f"Reranked {len(candidates)} candidates to {len(ranked_indices)}")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using cosine similarity")
                ranked_indices = ranked_indices[:top_k]
        else:
            ranked_indices = ranked_indices[:top_k]

        # Map back to original indices
        selected_indices = set()
        for pool_idx in ranked_indices:
            chunk = search_pool[pool_idx]
            original_idx = self.chunks.index(chunk)
            selected_indices.add(original_idx)

        # STEP 4: Context expansion
        expanded_indices = set()
        for idx in selected_indices:
            start = max(0, idx - context_window)
            end = min(len(self.chunks) - 1, idx + context_window)
            for i in range(start, end + 1):
                expanded_indices.add(i)

        # Return in narrative order
        result = [self.chunks[i] for i in sorted(expanded_indices)]
        
        logger.debug(f"Retrieved {len(selected_indices)} core + {len(expanded_indices) - len(selected_indices)} context chunks")
        return result
    
    def _filter_by_metadata(self, character_name: str = None) -> List[ChunkMetadata]:
        """
        Filter chunks by metadata (character mentions).
        """
        if not character_name:
            return self.chunks
        
        char_lower = character_name.lower()
        filtered = []
        
        for chunk in self.chunks:
            # Check if character mentioned in entities
            entities_lower = [e.lower() for e in chunk.entities]
            
            # Fuzzy match: check if character name is substring of any entity
            if any(char_lower in e or e in char_lower for e in entities_lower):
                filtered.append(chunk)
        
        logger.debug(f"Filtered {len(filtered)}/{len(self.chunks)} chunks for character '{character_name}'")
        return filtered
