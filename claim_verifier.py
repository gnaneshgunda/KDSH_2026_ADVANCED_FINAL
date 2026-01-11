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

        prompt = f"""Compare the CLAIM to the NARRATIVE EVIDENCE.
CLAIM: "{claim}"
EVIDENCE: {evidence_text}

Is the claim CONTRADICTED by the evidence?
A contradiction occurs if the evidence:
1. Explicitly states the opposite.
2. Establishes facts making the claim impossible (e.g., location, timing, or identity conflicts).
3. Violates character constraints established in the text.

Return JSON ONLY:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
  "explanation": "One sentence rationale for the verdict",
  "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.llm.chat(prompt)
            # Helper to clean markdown if LLM adds it
            json_str = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception:
            return {"verdict": "NOT_MENTIONED", "explanation": "Analysis inconclusive", "confidence": 0.4}