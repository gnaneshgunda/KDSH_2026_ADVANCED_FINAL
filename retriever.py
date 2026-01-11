import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from models import ChunkMetadata
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retrieve relevant narrative chunks with character-anchored boosting"""
    
    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])
        logger.info(f"HybridRetriever initialized with {len(chunks)} chunks")

    def retrieve(self, query: str, character_name: str = None, top_k: int = 5) -> List[ChunkMetadata]:
        """
        Retrieve chunks using vector similarity with a boost for character mentions.
        """
        if not self.chunks:
            return []
            
        try:
            query_embedding = np.array(self.client.embed([query])[0])
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Entity-Based Boosting: If character name is provided, boost chunks mentioning them
            if character_name and character_name.lower() != "unknown":
                char_lower = character_name.lower()
                for i, chunk in enumerate(self.chunks):
                    # We give a 15% similarity boost if the character name appears in the text
                    if char_lower in chunk.text.lower():
                        similarities[i] += 0.15 
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.chunks[int(i)] for i in top_indices if i < len(self.chunks)]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieve: {e}")
            return []