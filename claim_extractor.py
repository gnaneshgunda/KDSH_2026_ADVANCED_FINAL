"""
Claim extraction from backstory text
"""

import logging
from typing import List
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract atomic factual claims from backstory text"""

    def __init__(self, llm: NVIDIAClient):
        self.llm = llm
        logger.info("ClaimExtractor initialized")

    def extract_claims(self, backstory: str) -> List[str]:
        """
        Extract atomic factual claims from backstory text.
        
        Args:
            backstory: Backstory text (string or combined from dict)
            
        Returns:
            List of claim strings
        """
        if not backstory or not backstory.strip():
            logger.warning("Empty backstory provided to extract_claims")
            return []

        prompt = f""""
        ### ROLE
You are a Narrative Logic Auditor. Your task is to decompose a character backstory into unique, atomic, and verifiable "Narrative Anchors" to be checked against a source text.

### INPUT
BACKSTORY:
{backstory}

### EXTRACTION RULES (STRICT)
1. **Atomic Only**: Each claim must be a single, independent fact. No "and," "but," or "because."
2. **De-duplication**: Do not extract the same fact twice, even if phrased differently in the text.
3. **Focus on Anchors**: Prioritize "Hard Facts" that are likely to be contradicted if the backstory is fake:
    - **Identity**: Names, roles, professions.
    - **Spatiotemporal**: Specific locations (cities, rooms) and specific times (years, ages, durations).
    - **Relational**: Family ties, specific interactions with other characters.
    - **Causal/Action**: Specific events that happened (e.g., "John bought a car," not "John liked his car").
4. **Discard Fluff**: Ignore subjective descriptions (e.g., "He was very brave") unless they are core character traits mentioned in the narrative.

### OUTPUT FORMAT
Return ONLY a newline-separated list of claims. 
- No numbering.
- No introductory text.
- No formatting or punctuation at the start of lines.

### EXAMPLE
Input: "In 1995, Elara moved to London after her father, a clockmaker, died in a fire."
Output:
Elara moved to London in 1995
Elara's father was a clockmaker
Elara's father died in a fire

Claims:"""
        
        try:
            response = self.llm.chat(prompt)
            
            # Parse response: split by newlines and clean
            claims = [
                line.strip()
                for line in response.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*'))
            ]
            
            # Filter empty strings
            claims = [c for c in claims if c]
            
            logger.info(f"Extracted {len(claims)} claims from backstory")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []