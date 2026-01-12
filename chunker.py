"""
Paragraph-aware semantic chunker with NLP-based metadata extraction
"""

import logging
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from config import nlp

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunk narrative text with NLP-based metadata extraction using spaCy
    """

    def __init__(self, max_chunk_size: int = 400):
        self.max_chunk_size = max_chunk_size
        logger.info(f"SemanticChunker initialized (max_words={max_chunk_size})")

    def chunk_text(self, text: str, overlap_ratio: float = 0.2) -> List[Dict]:
        """
        Chunk text with NLP-based metadata extraction.
        
        Returns:
            List of dicts with keys: text, start_pos, end_pos, metadata
        """
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        char_position = 0
        
        for para in paragraphs:
            try:
                sentences = sent_tokenize(para)
            except:
                sentences = [s.strip() for s in para.split('.') if s.strip()]
            
            for sent in sentences:
                sent_words = len(sent.split())
                
                if current_size + sent_words > self.max_chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_start = char_position - len(chunk_text)
                    
                    metadata = self._extract_metadata(chunk_text)
                    
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': chunk_start,
                        'end_pos': char_position,
                        'metadata': metadata
                    })
                    
                    overlap_count = max(1, int(len(current_chunk) * overlap_ratio))
                    current_chunk = current_chunk[-overlap_count:]
                    current_size = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sent)
                current_size += sent_words
                char_position += len(sent) + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = self._extract_metadata(chunk_text)
            chunks.append({
                'text': chunk_text,
                'start_pos': char_position - len(chunk_text),
                'end_pos': char_position,
                'metadata': metadata
            })
        
        logger.info(f"Chunked text into {len(chunks)} segments (avg size: {sum(len(c['text'].split()) for c in chunks) / len(chunks) if chunks else 0:.0f} words)")
        return chunks
    
    def _extract_metadata(self, text: str) -> Dict:
        """
        Extract metadata using spaCy NLP:
        - characters: PERSON entities
        - temporal: DATE, TIME entities + temporal expressions
        - locations: GPE, LOC entities
        - has_dialogue: Contains quotes
        """
        doc = nlp(text)
        
        metadata = {
            'characters': [],
            'temporal': [],
            'locations': [],
            'has_dialogue': '"' in text or "'" in text or '"' in text
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and ent.text not in metadata['characters']:
                metadata['characters'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME'] and ent.text not in metadata['temporal']:
                metadata['temporal'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC'] and ent.text not in metadata['locations']:
                metadata['locations'].append(ent.text)
        
        # Extract temporal expressions using dependency parsing
        temporal_deps = ['npadvmod', 'tmod']
        for token in doc:
            if token.dep_ in temporal_deps and token.text.lower() not in metadata['temporal']:
                metadata['temporal'].append(token.text.lower())
        
        # Limit results
        metadata['characters'] = metadata['characters'][:10]
        metadata['temporal'] = metadata['temporal'][:5]
        metadata['locations'] = metadata['locations'][:5]
        
        return metadata
