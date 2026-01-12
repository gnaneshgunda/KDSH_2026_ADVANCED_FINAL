"""
Context vector builder with NLP-based analysis
"""

import logging
from typing import List
import numpy as np
from config import nlp

logger = logging.getLogger(__name__)


class ContextVectorBuilder:
    """
    Build context vectors using spaCy NLP for:
    - Sentiment analysis (via token polarity)
    - Temporal marker extraction (NER + dependency parsing)
    - Causal indicator extraction (dependency patterns)
    """

    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        logger.info(f"ContextVectorBuilder initialized (embedding_dim={embedding_dim})")

    def build_context_vector(self, text: str, embedding: np.ndarray) -> np.ndarray:
        """Build enriched context vector."""
        return embedding

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using spaCy token-level analysis.
        """
        doc = nlp(text)
        
        # Simple sentiment based on adjective and verb polarity
        positive_pos = {'ADJ', 'VERB', 'ADV'}
        positive_words = {'good', 'great', 'happy', 'love', 'wonderful', 'excellent', 'beautiful', 'proud', 'successful', 'amazing'}
        negative_words = {'bad', 'terrible', 'sad', 'hate', 'awful', 'horrible', 'tragic', 'failed', 'angry', 'afraid'}
        
        pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
        neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return float(np.clip((pos_count - neg_count) / total, -1.0, 1.0))

    def extract_temporal_markers(self, text: str) -> List[str]:
        """
        Extract temporal markers using NER and dependency parsing.
        """
        doc = nlp(text)
        markers = []
        
        # Extract DATE and TIME entities
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                markers.append(ent.text)
        
        # Extract temporal adverbials via dependency
        for token in doc:
            if token.dep_ in ['npadvmod', 'tmod'] and token.pos_ in ['NOUN', 'ADV']:
                markers.append(token.text.lower())
        
        return list(set(markers))[:5]

    def extract_causal_indicators(self, text: str) -> List[str]:
        """
        Extract causal relationships using dependency parsing.
        """
        doc = nlp(text)
        indicators = []
        
        # Look for causal markers
        causal_markers = {'because', 'since', 'as', 'due', 'caused', 'result', 'led', 'therefore', 'thus', 'hence'}
        
        for token in doc:
            if token.text.lower() in causal_markers:
                indicators.append(token.text.lower())
            # Look for causal dependencies
            elif token.dep_ in ['advcl', 'mark'] and token.head.pos_ == 'VERB':
                indicators.append(f"{token.text}->{token.head.text}")
        
        return indicators[:5]

    def analyze_entities(self, text: str) -> List[str]:
        """
        Extract named entities using spaCy NER.
        """
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
        return list(set(entities))[:10]
