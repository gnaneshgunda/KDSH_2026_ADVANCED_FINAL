"""
Claim verification against narrative evidence with improved reasoning
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
            return {"verdict": "NOT_MENTIONED", "rationale": "No evidence available", "confidence": 0.5}

        evidence_text = "\n\n".join(f"[Evidence {i+1}]\n{e}" for i, e in enumerate(evidence_chunks))

        prompt = f"""You are a precise fact-checker analyzing whether a CLAIM about a character's backstory is supported by NARRATIVE EVIDENCE.

**CLAIM TO VERIFY:**
"{claim}"

**NARRATIVE EVIDENCE:**
{evidence_text}

**VERIFICATION RULES:**

1. **SUPPORTED** - Use ONLY if:
   - The evidence EXPLICITLY states the claim or a direct synonym
   - Example: Claim "He was born in Paris" + Evidence "Jean was born in Paris" = SUPPORTED
   - Direct paraphrases count ("enormous" = "huge")

2. **CONTRADICTED** - Use ONLY if:
   - The evidence EXPLICITLY contradicts the claim with opposite facts
   - Example: Claim "He was in London" + Evidence "He was in Paris at that time" = CONTRADICTED
   - Logical impossibilities count (timeline conflicts, mutually exclusive events)

3. **NOT_MENTIONED** - Use if:
   - The evidence is silent on the specific claim
   - The evidence is thematically related but doesn't confirm the specific fact
   - You need to infer or assume to connect evidence to claim
   - Example: Claim "He was sad" + Evidence "He was crying" = NOT_MENTIONED (crying doesn't explicitly state sadness)

**CRITICAL:**
- Absence of evidence is NOT contradiction
- Thematic consistency is NOT support
- Reasonable inferences are NOT support
- Only EXPLICIT statements count

**OUTPUT FORMAT (JSON only, no markdown):**
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
  "rationale": "Quote the specific evidence that led to this verdict. If NOT_MENTIONED, explain what specific detail is missing.",
  "confidence": 0.0-1.0
}}
"""
        try:
            response = self.llm.chat(prompt)
            json_str = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(json_str)
            
            # Ensure rationale is detailed
            if 'rationale' not in result or len(result['rationale']) < 20:
                result['rationale'] = f"Verdict: {result.get('verdict', 'UNKNOWN')}. Analysis incomplete."
            
            return result
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {"verdict": "NOT_MENTIONED", "rationale": f"Analysis error: {str(e)}", "confidence": 0.4}