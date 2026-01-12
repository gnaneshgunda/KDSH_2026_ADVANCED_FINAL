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

        prompt = f"""
You are a strict logic engine comparing a proposed BACKSTORY CLAIM against established NARRATIVE EVIDENCE. Use a step-by-step Chain of Thought process to ensure logical rigor.

### INPUTS
CLAIM: "{claim}"
EVIDENCE: "{evidence_text}"

### LOGIC RULES
1. **Burden of Proof**: Use ONLY the provided EVIDENCE. External knowledge, common sense assumptions, or "hallucinated" context are strictly forbidden.
2. **The "Silence" Rule**: If the evidence does not explicitly address the specific subject matter, the verdict is NOT_MENTIONED. Being "consistent with the theme" or "thematically plausible" is NOT enough for SUPPORTED.
3. **The "Conflict" Rule**: To be CONTRADICTED, there must be a direct factual clash or a logical impossibility (e.g., the evidence says a character is in London, the claim says they are in New York at the exact same time).
4. **Synonym vs. Inference**: Synonyms (e.g., "enormous" vs. "huge") are SUPPORTED. Inferences (e.g., "he was crying" implies "he was sad") are NOT_MENTIONED unless the emotion is explicitly stated.

### EVALUATION PROCESS (Chain of Thought)
Follow these steps internally before providing the verdict:
- Step 1: Identify the core subjects and actions in the CLAIM.
- Step 2: Locate the specific segments in the EVIDENCE that discuss these subjects/actions.
- Step 3: Check for a direct match (Supported).
- Step 4: Check for a direct contradiction or mutual exclusivity (Contradicted).
- Step 5: If neither 3 nor 4 applies, or if the evidence is silent on a key detail of the claim, default to Not Mentioned.

### OUTPUT FORMAT
Return valid JSON only. No markdown formatting.
Return valid JSON only. No markdown formatting.
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
  "rationale": "Detailed explanation...",
  "confidence": 0.0-1.0
}}
"""
        try:
            response = self.llm.chat(prompt)
            # Helper to clean markdown if LLM adds it
            json_str = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception:
            return {"verdict": "NOT_MENTIONED", "explanation": "Analysis inconclusive", "confidence": 0.4}