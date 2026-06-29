"""
Claim verification against narrative evidence with chain-of-thought reasoning
and confidence calibration.
"""

import logging
import json
import re
from typing import List, Dict
from nvidia_client import NVIDIAClient

logger = logging.getLogger(__name__)


def _calibrate_confidence(verdict: str, raw_confidence: float, quoted_evidence: str) -> float:
    """Map verdict + evidence quality to calibrated confidence ranges.

    Ranges:
      SUPPORTED  + direct quote        → 0.85 – 0.95
      SUPPORTED  + paraphrase          → 0.70 – 0.85
      CONTRADICTED + direct conflict   → 0.85 – 0.95
      CONTRADICTED + paraphrase        → 0.70 – 0.85
      NOT_MENTIONED                    → 0.40 – 0.60
    """
    has_quote = bool(quoted_evidence and len(quoted_evidence.strip()) > 10)

    if verdict == "SUPPORTED":
        if has_quote:
            return max(0.85, min(0.95, raw_confidence))
        else:
            return max(0.70, min(0.85, raw_confidence))
    elif verdict == "CONTRADICTED":
        if has_quote:
            return max(0.85, min(0.95, raw_confidence))
        else:
            return max(0.70, min(0.85, raw_confidence))
    else:  # NOT_MENTIONED / UNKNOWN
        return max(0.40, min(0.60, raw_confidence))


class ClaimVerifier:
    """Verify individual claims against narrative evidence using chain-of-thought reasoning."""

    def __init__(self, llm_client: NVIDIAClient):
        self.llm = llm_client
        logger.info("ClaimVerifier initialized")

    def verify(self, claim: str, evidence_chunks: List[str]) -> Dict:
        if not evidence_chunks:
            return {
                "verdict": "NOT_MENTIONED",
                "rationale": "No evidence available",
                "confidence": 0.5,
                "quoted_evidence": "",
            }

        evidence_text = "\n\n".join(
            f"[Evidence {i+1}]\n{e}" for i, e in enumerate(evidence_chunks)
        )

        prompt = f"""You are a precise fact-checker verifying whether a CLAIM about a character's backstory is supported by NARRATIVE EVIDENCE. You MUST reason step-by-step.

**CLAIM TO VERIFY:**
"{claim}"

**NARRATIVE EVIDENCE:**
{evidence_text}

**INSTRUCTIONS — follow these four steps exactly:**

**Step 1 — Quote:** Copy the MOST RELEVANT sentence or phrase from the evidence above. If no evidence is relevant, write "No relevant passage found."

**Step 2 — Identify:** In one sentence, state what the quoted evidence explicitly says.

**Step 3 — Compare:** Are the claim and the evidence stating the SAME fact, DIFFERENT (unrelated) facts, or CONTRADICTORY facts? Explain briefly.

**Step 4 — Verdict:** Based ONLY on Step 3, assign one of:
  • SUPPORTED — evidence explicitly confirms the claim (direct statement or direct synonym/paraphrase).
  • CONTRADICTED — evidence explicitly states the opposite or a logically incompatible fact.
  • NOT_MENTIONED — evidence is silent on this specific claim, or requires inference to connect.

**VERIFICATION RULES (CRITICAL):**
- Absence of evidence is NOT contradiction.
- Thematic consistency is NOT support.
- Reasonable inferences are NOT support — only EXPLICIT statements count.
- Paraphrases and direct synonyms DO count as explicit ("enormous" = "huge").

**OUTPUT FORMAT (JSON only, no markdown, no extra text):**
{{
  "step1_quote": "<exact text from evidence or 'No relevant passage found'>",
  "step2_evidence_says": "<one-sentence summary>",
  "step3_comparison": "<same / different / contradictory — with brief explanation>",
  "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
  "quoted_evidence": "<the key phrase that most directly supports or contradicts the claim, or empty string>",
  "rationale": "<one-paragraph summary of reasoning>",
  "confidence": 0.0-1.0
}}
"""
        try:
            response = self.llm.chat(prompt)

            if not response or not response.strip():
                logger.warning("Empty response from LLM")
                return {
                    "verdict": "NOT_MENTIONED",
                    "rationale": "LLM returned empty response",
                    "confidence": 0.5,
                    "quoted_evidence": "",
                }

            # Clean response — remove markdown code blocks if present
            json_str = (
                response.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            # Try to parse JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as je:
                logger.error(
                    f"JSON decode error: {je}. Response was: {json_str[:200]}"
                )
                return {
                    "verdict": "NOT_MENTIONED",
                    "rationale": f"Unable to parse LLM response. Raw output: {json_str[:100]}",
                    "confidence": 0.4,
                    "quoted_evidence": "",
                }

            # Validate required fields
            if "verdict" not in result:
                logger.warning(f"Missing 'verdict' field in response: {result}")
                result["verdict"] = "NOT_MENTIONED"

            # Normalise verdict
            result["verdict"] = result["verdict"].upper().strip()
            if result["verdict"] not in ("SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"):
                result["verdict"] = "NOT_MENTIONED"

            # Build rationale from CoT steps if available
            cot_parts = []
            if result.get("step1_quote"):
                cot_parts.append(f"Quote: {result['step1_quote']}")
            if result.get("step2_evidence_says"):
                cot_parts.append(f"Evidence says: {result['step2_evidence_says']}")
            if result.get("step3_comparison"):
                cot_parts.append(f"Comparison: {result['step3_comparison']}")

            if "rationale" not in result or len(result.get("rationale", "")) < 20:
                if cot_parts:
                    result["rationale"] = " | ".join(cot_parts)
                else:
                    result["rationale"] = (
                        f"Verdict: {result.get('verdict', 'UNKNOWN')}. Analysis incomplete."
                    )

            # Ensure quoted_evidence exists
            if "quoted_evidence" not in result:
                result["quoted_evidence"] = result.get("step1_quote", "")

            # Confidence calibration
            raw_conf = float(result.get("confidence", 0.5))
            result["confidence"] = _calibrate_confidence(
                result["verdict"], raw_conf, result.get("quoted_evidence", "")
            )

            return result

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verdict": "NOT_MENTIONED",
                "rationale": f"Analysis error: {str(e)}",
                "confidence": 0.4,
                "quoted_evidence": "",
            }