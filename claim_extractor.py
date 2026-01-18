"""
Claim extraction from backstory text using NLP
"""

import logging
from typing import List
from nvidia_client import NVIDIAClient
from config import nlp

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract atomic factual claims from backstory text using NLP"""

    def __init__(self, llm: NVIDIAClient):
        self.llm = llm
        logger.info("ClaimExtractor initialized")

    def extract_claims(self, backstory: str) -> List[str]:
        """
        Extract atomic factual claims from backstory text.
        Uses NLP to identify key entities and relations before LLM extraction.
        
        Args:
            backstory: Backstory text
            
        Returns:
            List of claim strings
        """
        if not backstory or not backstory.strip():
            logger.warning("Empty backstory provided to extract_claims")
            return []

        # Pre-process with NLP to identify key entities
        doc = nlp(backstory)
        
        # Extract key entities for context
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        
        entity_context = ""
        if persons:
            entity_context += f"\nKey characters: {', '.join(set(persons[:5]))}"
        if dates:
            entity_context += f"\nKey dates: {', '.join(set(dates[:5]))}"
        if locations:
            entity_context += f"\nKey locations: {', '.join(set(locations[:5]))}"

        prompt = f"""### ROLE
You are a Narrative Logic Auditor. Your task is to decompose a character backstory into unique, atomic, and verifiable "Narrative Anchors" to be checked against a source text.

### INPUT
BACKSTORY:
{backstory}
{entity_context}

### EXTRACTION RULES (STRICT)
1. **Atomic Only**: Each claim must be a single, independent fact. No "and," "but," or "because."
2. **De-duplication**: Do not extract the same fact twice, even if phrased differently in the text.
3. **Focus on Anchors**: Prioritize "Hard Facts" that are likely to be contradicted if the backstory is fake:
    - **Identity**: Names, roles, professions.
    - **Spatiotemporal**: Specific locations (cities, rooms) and specific times (years, ages, durations).
    - **Relational**: Family ties, specific interactions with other characters.
    - **Causal/Action**: Specific events that happened (e.g., "John bought a car," not "John liked his car").
4. **Discard Fluff**: Ignore subjective descriptions (e.g., "He was very brave") unless they are core character traits mentioned in the narrative.
5. **Entity-Focused**: Ensure claims reference the key entities identified above.

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
            
            # Check if response is a JSON error from stub
            if response.strip().startswith('{') and '"error"' in response:
                logger.warning("Stub fallback detected in claim extraction. Using simple sentence splitting.")
                # Fallback: extract sentences as claims
                doc = nlp(backstory)
                claims = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
                return claims[:12]  # Limit to 12
            
            # Parse response: split by newlines and clean
            claims = [
                line.strip()
                for line in response.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*', '{', '}'))
            ]
            
            # Filter empty strings and validate claims contain entities
            validated_claims = []
            for claim in claims:
                # Skip JSON-like strings
                if claim.startswith(('{', '}')):
                    continue
                    
                if claim and self._validate_claim(claim, persons, dates, locations):
                    validated_claims.append(claim)
            
            logger.info(f"Extracted {len(validated_claims)} validated claims from backstory")
            return validated_claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            # Fallback: use simple sentence extraction
            try:
                doc = nlp(backstory)
                claims = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
                logger.info(f"Fallback extraction: {len(claims[:12])} claims from sentences")
                return claims[:12]
            except:
                return []
    
    def _validate_claim(self, claim: str, persons: List[str], dates: List[str], locations: List[str]) -> bool:
        """
        Validate that claim contains at least one key entity or is substantive.
        """
        claim_lower = claim.lower()
        
        # Check if claim contains any key entities
        for person in persons:
            if person.lower() in claim_lower:
                return True
        for date in dates:
            if date.lower() in claim_lower:
                return True
        for location in locations:
            if location.lower() in claim_lower:
                return True
        
        # If no entities, check if claim is substantive (>5 words)
        return len(claim.split()) > 5
