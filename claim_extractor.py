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

        prompt = f"""Extract ALL atomic factual claims from the text below. Each claim should be a single, verifiable fact.

Rules:
- One claim = one fact (no compound claims)
- Be specific and concrete
- No inferences or assumptions
- No explanations or reasoning
- Include temporal, causal, and emotional facts

Backstory:
{backstory}

Output format:
Return ONLY a newline-separated list of claims, one per line. Start with a claim immediately, no numbering.
Example:
John was born in 1980
John's father was a lawyer
John had a fear of water

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