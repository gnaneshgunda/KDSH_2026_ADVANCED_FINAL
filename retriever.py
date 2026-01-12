import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from models import ChunkMetadata
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Two-step retrieval:
    1. Metadata-based filtering (character-aware)
    2. Semantic ranking (cosine similarity)
    3. Context expansion (+/- N neighboring chunks)
    """

    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])

    def retrieve(
        self,
        query: str,
        character_name: str = None,
        top_k: int = 7,
        context_window: int = 2
    ) -> List[ChunkMetadata]:

        if not self.chunks:
            return []

        # ---- STEP 1: metadata-based character filtering ----
        if character_name:
            char_chunks = [
                c for c in self.chunks
                if character_name.lower() in [e.lower() for e in c.entities]
            ]
        else:
            char_chunks = []

        # Prefer character-specific pool if sufficient
        search_pool = char_chunks if len(char_chunks) >= top_k else self.chunks

        # ---- STEP 2: semantic similarity ranking ----
        pool_embeddings = np.array([c.embedding for c in search_pool])
        query_emb = np.array(self.client.embed([query])[0])

        sims = cosine_similarity([query_emb], pool_embeddings)[0]
        ranked_pool_indices = np.argsort(sims)[::-1][:top_k]

        # Map selected chunks back to ORIGINAL indices
        selected_indices = set()
        for pool_idx in ranked_pool_indices:
            chunk = search_pool[pool_idx]
            original_idx = self.chunks.index(chunk)
            selected_indices.add(original_idx)

        # ---- STEP 3: context expansion (+/- context_window) ----
        expanded_indices = set()
        for idx in selected_indices:
            start = max(0, idx - context_window)
            end = min(len(self.chunks) - 1, idx + context_window)
            for i in range(start, end + 1):
                expanded_indices.add(i)

        # ---- STEP 4: return in narrative order ----
        expanded_chunks = [self.chunks[i] for i in sorted(expanded_indices)]

        logger.debug(
            f"Retrieved {len(selected_indices)} core chunks, "
            f"expanded to {len(expanded_chunks)} using ±{context_window} window"
        )

        return expanded_chunks
