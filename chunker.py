"""
Paragraph-aware semantic chunker with rich metadata extraction
"""

import logging
import re
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunk narrative text with metadata extraction:
    - Characters mentioned
    - Temporal markers (dates, time periods)
    - Location references
    - Dialogue vs narrative
    """

    def __init__(self, max_chunk_size: int = 400):
        self.max_chunk_size = max_chunk_size
        
        # Temporal patterns
        self.temporal_patterns = [
            r'\b(\d{4})\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(spring|summer|autumn|fall|winter)\b',
            r'\b(morning|afternoon|evening|night|dawn|dusk)\b',
            r'\b(yesterday|today|tomorrow|ago|later|before|after|during|while)\b',
            r'\b(childhood|youth|boyhood|adolescence|adulthood)\b'
        ]
        
        # Location patterns
        self.location_patterns = [
            r'\b(in|at|near|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(city|town|village|country|island|mountain|river|sea|ocean|forest|desert)\b'
        ]
        
        logger.info(f"SemanticChunker initialized (max_words={max_chunk_size})")

    def chunk_text(self, text: str, overlap_ratio: float = 0.2) -> List[Dict]:
        """
        Chunk text with metadata extraction.
        
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
                    
                    # Extract metadata
                    metadata = self._extract_metadata(chunk_text)
                    
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': chunk_start,
                        'end_pos': char_position,
                        'metadata': metadata
                    })
                    
                    # Overlap
                    overlap_count = max(1, int(len(current_chunk) * overlap_ratio))
                    current_chunk = current_chunk[-overlap_count:]
                    current_size = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sent)
                current_size += sent_words
                char_position += len(sent) + 1
        
        # Flush remaining
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
        Extract metadata from chunk:
        - characters: Capitalized names
        - temporal: Time references
        - locations: Place names
        - has_dialogue: Contains quotes
        """
        metadata = {
            'characters': [],
            'temporal': [],
            'locations': [],
            'has_dialogue': '"' in text or "'" in text
        }
        
        # Extract characters (capitalized words, filter common words)
        words = text.split()
        common_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 'But', 'Or', 'As', 'If', 'When', 'Where', 'Why', 'How', 'What', 'Which', 'Who', 'Whom', 'This', 'That', 'These', 'Those', 'I', 'He', 'She', 'It', 'We', 'They', 'My', 'His', 'Her', 'Its', 'Our', 'Their'}
        
        for word in words:
            clean_word = word.strip('.,!?;:"\'()')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2 and clean_word not in common_words:
                if clean_word not in metadata['characters']:
                    metadata['characters'].append(clean_word)
        
        # Limit to top 10
        metadata['characters'] = metadata['characters'][:10]
        
        # Extract temporal markers
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata['temporal'].extend(matches)
        metadata['temporal'] = list(set(metadata['temporal']))[:5]
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                metadata['locations'].extend([m[1] if len(m) > 1 else m[0] for m in matches])
            else:
                metadata['locations'].extend(matches)
        metadata['locations'] = list(set(metadata['locations']))[:5]
        
        return metadata