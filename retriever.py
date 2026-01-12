import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from models import ChunkMetadata
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Two-step retrieval: Entity Filtering followed by Semantic Ranking"""
    
    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])

    def retrieve(self, query: str, character_name: str = None, top_k: int = 7) -> List[ChunkMetadata]:
        if not self.chunks: return []
        
        # Use metadata stored in .pkl to filter for character presence 
        char_chunks = [c for c in self.chunks if character_name.lower() in [e.lower() for e in c.entities]]
        
        # If character scenes exist, search there first to avoid global noise 
        search_pool = char_chunks if len(char_chunks) >= top_k else self.chunks
        
        pool_embeddings = np.array([c.embedding for c in search_pool])
        query_emb = np.array(self.client.embed([query])[0])
        sims = cosine_similarity([query_emb], pool_embeddings)[0]
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [search_pool[i] for i in top_indices]