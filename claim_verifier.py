"""
Claim verification against narrative evidence
"""

import logging
import json
from typing import List, Dict
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)


class ClaimVerifier:
    """Verify individual claims against narrative evidence"""

    def __init__(self, llm_client: NVIDIAClient):
        self.llm = llm_client
        logger.info("ClaimVerifier initialized")

    def verify(self, claim: str, evidence_chunks: List[str]) -> Dict:
        if not evidence_chunks:
            return {"verdict": "NOT_MENTIONED", "explanation": "No evidence available", "confidence": 0.5}

        evidence_text = "\n\n".join(f"[Chunk {i+1}] {e}" for i, e in enumerate(evidence_chunks))

        prompt = f"""Compare the BACKSTORY CLAIM to the NARRATIVE EVIDENCE.
CLAIM: "{claim}"
EVIDENCE: {evidence_text}

Task: Determine if this claim is logically incompatible with the narrative events, character states, or world-rules[cite: 8, 39].
Rules:
1. CONTRADICTED (0): The evidence rules out the claim (e.g., character's location or status conflicts)[cite: 6, 12].
2. SUPPORTED: Explicit evidence confirms the claim.
3. NOT_MENTIONED: Narrative is silent/neutral. Silence is NOT a contradiction[cite: 36, 151].

Return JSON ONLY: {{"verdict": "SUPPORTED"|"CONTRADICTED"|"NOT_MENTIONED", "explanation": "Rationale", "confidence": 0.0-1.0}}"""
        
        try:
            response = self.llm.chat(prompt)
            # Helper to clean markdown if LLM adds it
            json_str = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception:
            return {"verdict": "NOT_MENTIONED", "explanation": "Analysis inconclusive", "confidence": 0.4}