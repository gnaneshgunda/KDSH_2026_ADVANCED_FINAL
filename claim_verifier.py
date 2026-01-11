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
        """
        Verify a claim against narrative evidence.
        
        Args:
            claim: The claim to verify
            evidence_chunks: List of narrative text chunks as evidence
            
        Returns:
            Dict with keys: verdict, explanation, confidence
            verdict: SUPPORTED | CONTRADICTED | NOT_MENTIONED | UNKNOWN
        """
        if not evidence_chunks:
            logger.warning(f"No evidence provided for claim: {claim}")
            return {
                "verdict": "NOT_MENTIONED",
                "explanation": "No evidence available in narrative",
                "confidence": 0.5
            }

        prompt = self._build_prompt(claim, evidence_chunks)
        
        try:
            response = self.llm.chat(prompt)
            result = self._parse_response(response, claim)
            logger.debug(f"Claim verification result: {result['verdict']}")
            return result
            
        except Exception as e:
            logger.error(f"Error verifying claim: {e}")
            return {
                "verdict": "UNKNOWN",
                "explanation": f"Verification error: {str(e)}",
                "confidence": 0.3
            }

    def _build_prompt(self, claim: str, evidence: List[str]) -> str:
        """Build prompt for claim verification"""
        evidence_text = "\n\n".join(
            f"[Evidence {i+1}]\n{e}" for i, e in enumerate(evidence[:5])
        )

        return f"""You are a factual consistency checker. Your task is to determine if the given claim is supported, contradicted, or not mentioned in the narrative evidence.

CLAIM:
"{claim}"

NARRATIVE EVIDENCE:
{evidence_text}

TASK: Determine whether the claim is:
- SUPPORTED: The narrative clearly supports or strongly implies this claim
- CONTRADICTED: The narrative explicitly contradicts or refutes this claim
- NOT_MENTIONED: The narrative does not address this claim at all

RULES:
- Base your judgment ONLY on the provided evidence
- Do NOT use external knowledge
- Be conservative: implications must be strong and clear
- If evidence is insufficient, choose NOT_MENTIONED
- A claim is only CONTRADICTED if the narrative explicitly opposes it

Return a JSON response with ONLY these fields (no markdown, no extra text):
{{
  "verdict": "SUPPORTED" or "CONTRADICTED" or "NOT_MENTIONED",
  "explanation": "One sentence justification",
  "confidence": 0.0-1.0
}}

Response:"""

    def _parse_response(self, response: str, claim: str) -> Dict:
        """
        Parse LLM response to extract verdict, explanation, confidence.
        
        Handles both JSON and text-based responses.
        """
        response = response.strip()
        
        # Try JSON parsing first
        try:
            result = json.loads(response)
            verdict = result.get("verdict", "UNKNOWN").upper()
            explanation = result.get("explanation", "No explanation provided")
            confidence = float(result.get("confidence", 0.5))
            
            # Validate verdict
            if verdict not in ["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]:
                verdict = "UNKNOWN"
            
            return {
                "verdict": verdict,
                "explanation": explanation,
                "confidence": min(1.0, max(0.0, confidence))
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Fallback: parse text response for verdict keywords
        response_upper = response.upper()
        verdict = "UNKNOWN"
        confidence = 0.5
        
        if "CONTRADICTED" in response_upper:
            verdict = "CONTRADICTED"
            confidence = 0.8
        elif "SUPPORTED" in response_upper or "SUPPORTS" in response_upper:
            verdict = "SUPPORTED"
            confidence = 0.8
        elif "NOT_MENTIONED" in response_upper or "NOT MENTIONED" in response_upper:
            verdict = "NOT_MENTIONED"
            confidence = 0.7
        
        logger.warning(f"Fallback verdict parsing for claim: {claim[:50]}...")
        
        return {
            "verdict": verdict,
            "explanation": response[:300],
            "confidence": confidence
        }