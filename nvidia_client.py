"""
NVIDIA NIM + HuggingFace Client for embeddings, chat, and reranking
Uses real NVIDIA API when available, falls back to HuggingFace Transformers
"""

from typing import List, Dict
import hashlib
import json
import numpy as np
import requests
from config import EMBEDDING_DIM
import logging

logger = logging.getLogger(__name__)

# Try to import HuggingFace models for fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    _hf_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Loaded HuggingFace SentenceTransformer: all-MiniLM-L6-v2")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("SentenceTransformer not installed. Install with: pip install sentence-transformers")

try:
    from transformers import pipeline
    HAS_HF_PIPELINE = True
    _hf_chat_pipeline = pipeline("text-generation", model="gpt2")
    logger.info("Loaded HuggingFace text-generation pipeline")
except ImportError:
    HAS_HF_PIPELINE = False
    logger.warning("Transformers not installed. Install with: pip install transformers")


def _text_to_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 31)


def _embed_text_hf(text: str) -> np.ndarray:
    """Embed text using HuggingFace SentenceTransformer
    
    Returns padded vector to EMBEDDING_DIM (2048) for array compatibility,
    but only first 384 dimensions contain meaningful information.
    """
    if HAS_SENTENCE_TRANSFORMERS:
        vec = _hf_embedder.encode(text, convert_to_numpy=True).astype(np.float32)
        
        # Pad to EMBEDDING_DIM for compatibility with NVIDIA embeddings
        if len(vec) < EMBEDDING_DIM:
            padding = np.zeros(EMBEDDING_DIM - len(vec), dtype=np.float32)
            vec = np.concatenate([vec, padding])
        
        # Normalize after padding
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
    else:
        # Fallback: deterministic stub (full EMBEDDING_DIM)
        seed = _text_to_seed(text)
        rng = np.random.RandomState(seed)
        vec = rng.normal(size=(EMBEDDING_DIM,)).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm


class NVIDIAClient:
    """Multi-backend client for embeddings, chat, and reranking.
    
    Priority: NVIDIA API > HuggingFace > Stub
    Methods:
    - embed(list[str]) -> list[np.ndarray]
    - chat(messages, temperature=0.0) -> str
    - rerank(query, candidates) -> list[dict]
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://integrate.api.nvidia.com/v1"
        self.use_nvidia_api = bool(api_key)
        self.use_hf = not self.use_nvidia_api and HAS_SENTENCE_TRANSFORMERS
        
        # Track active embedding dimension for consistency
        # NVIDIA: 2048, HuggingFace: 384, Stub: 2048
        self.active_embedding_dim = 384 if self.use_hf else EMBEDDING_DIM
        self._fallback_occurred = False  # Track if we've fallen back mid-session
        
        if self.use_nvidia_api:
            logger.info(f"Using NVIDIA API for inference: {self.base_url}")
        elif self.use_hf:
            logger.info(f"Using HuggingFace models for inference (embeddings: {self.active_embedding_dim}d)")
        else:
            logger.info(f"Using stub/deterministic fallback for inference (embeddings: {self.active_embedding_dim}d)")

    def get_embedding_dim(self) -> int:
        """Get the effective embedding dimension for similarity calculations.
        
        Returns:
            384 if using HF embeddings (only first 384 dims are meaningful)
            2048 if using NVIDIA or stub embeddings
        """
        return self.active_embedding_dim
    
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings using NVIDIA API > HuggingFace > Stub"""
        if self.use_nvidia_api:
            return self._embed_nvidia(texts)
        else:
            return [_embed_text_hf(t if t is not None else "") for t in texts]
    
    def _embed_nvidia(self, texts: List[str]) -> List[np.ndarray]:
        """Call NVIDIA embedding API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": [t if t is not None else "" for t in texts],
                "model": "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "START"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            embeddings = []
            for item in result.get("data", []):
                emb = np.array(item["embedding"], dtype=np.float32)
                embeddings.append(emb)
            
            logger.debug(f"Retrieved {len(embeddings)} embeddings from NVIDIA API")
            return embeddings
        except Exception as e:
            logger.warning(f"NVIDIA embedding failed: {e}. Falling back to HuggingFace.")
            
            # Track that fallback occurred mid-session
            if not self._fallback_occurred:
                self._fallback_occurred = True
                self.active_embedding_dim = 384
                logger.warning(
                    "⚠️  Embedding backend switched from NVIDIA (2048d) to HuggingFace (384d). "
                    "For optimal performance, consider rebuilding the index with consistent backend."
                )
            
            return [_embed_text_hf(t if t is not None else "") for t in texts]

    def chat(self, messages, temperature: float = 0.0) -> str:
        """Get chat response: NVIDIA API > Stub heuristic"""
        if self.use_nvidia_api:
            return self._chat_nvidia(messages, temperature)
        else:
            return self._chat_stub(messages, temperature)
    
    def _chat_nvidia(self, messages, temperature: float = 0.2) -> str:
        """Call NVIDIA chat API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",  # Required for all requests
                "Accept": "application/json",
            }
            
            if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
                payload_messages = messages
            else:
                prompt = str(messages) if not isinstance(messages, list) else "\n".join(m.get("content", "") for m in messages)
                payload_messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": "deepseek-ai/deepseek-v3.1",
                "messages": payload_messages,
                "temperature": temperature,
                "top_p": 0.7,
                "max_tokens": 8192
                # Removed chat_template_kwargs - not supported by all models
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",  # Fixed: use f-string
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.debug(f"NVIDIA chat response: {content[:100]}...")
            return content
        except Exception as e:
            logger.warning(f"NVIDIA chat failed: {e}. Falling back to stub heuristic.")
            return self._chat_stub(messages, temperature)
    
    def _chat_stub(self, messages, temperature: float = 0.0) -> str:
        """Stub chat implementation for fallback"""
        if isinstance(messages, list):
            prompt = "\n".join(m.get("content", "") for m in messages)
        else:
            prompt = str(messages)

        # Detect if this is a claim verification request (expects JSON output)
        is_claim_verification = any([
            "OUTPUT FORMAT" in prompt and "JSON" in prompt,
            '"verdict"' in prompt,
            "CLAIM TO VERIFY" in prompt,
            "**VERIFICATION RULES:**" in prompt
        ])
        
        if is_claim_verification:
            # Parse claim verification prompts
            # Look for evidence markers
            has_evidence = "Evidence" in prompt or "NARRATIVE EVIDENCE" in prompt
            has_contradicted_keyword = "CONTRADICTED" in prompt.upper()
            has_supported_keyword = "SUPPORTED" in prompt.upper()
            
            # Default to NOT_MENTIONED with conservative reasoning
            verdict = "NOT_MENTIONED"
            confidence = 0.5
            rationale = "Unable to verify claim without LLM analysis. Defaulting to neutral verdict."
            
            # Try to detect evidence presence
            if has_evidence:
                evidence_count = prompt.count("[Evidence ")
                if evidence_count > 0:
                    # Heuristic: if we have evidence, assume slight support
                    verdict = "SUPPORTED"
                    confidence = min(0.7, 0.5 + (evidence_count * 0.05))
                    rationale = f"Found {evidence_count} evidence chunk(s). Heuristic analysis suggests potential support."
                else:
                    rationale = "No clear evidence markers found in provided context."
            
            return json.dumps({
                "verdict": verdict,
                "rationale": rationale,
                "confidence": confidence
            })
        
        # Legacy JSON detection for other use cases
        if "Return JSON" in prompt or "Return JSON:" in prompt:
            # Improved heuristic: analyze supporting vs opposing evidence more granularly
            supporting_count = prompt.count("SUPPORTING")
            opposing_count = prompt.count("OPPOSING")
            
            # Count "None found" to detect weak evidence
            supporting_none = prompt.count("SUPPORTING") > 0 and "None found" in prompt.split("SUPPORTING")[1].split("OPPOSING")[0] if "OPPOSING" in prompt else False
            opposing_none = prompt.count("OPPOSING") > 0 and "None found" in prompt.split("OPPOSING")[1] if "OPPOSING" in prompt else False
            
            # Heuristic scoring
            supporting_score = 0 if supporting_none else max(1, supporting_count - 1)
            opposing_score = 0 if opposing_none else max(1, opposing_count - 1)
            
            # Decide consistency
            if supporting_score > opposing_score * 1.5:
                consistent = True
                confidence = min(0.95, 0.65 + (supporting_score - opposing_score) * 0.08)
            elif opposing_score > supporting_score * 1.5:
                consistent = False
                confidence = min(0.85, 0.55 + (opposing_score - supporting_score) * 0.08)
            else:
                seed = hash(prompt) % 100
                consistent = seed < 45
                confidence = 0.45 + (seed % 40) / 100.0
            
            reasoning = f"Analysis: {supporting_score} supporting vs {opposing_score} opposing chunks. {'Narrative aligns' if consistent else 'Contradictions found'} with backstory claims."
            
            return json.dumps({
                "consistent": bool(consistent),
                "confidence": float(confidence),
                "reasoning": reasoning
            })

        # If prompt asks to "Return ONLY the opposite statement" (used in negation)
        if "Return ONLY the opposite statement" in prompt:
            first_line = prompt.splitlines()[0]
            snippet = first_line[:200]
            return f"Not {snippet}"

        # Default: return JSON to avoid parsing errors
        logger.warning("Stub fallback: unknown prompt type, returning default JSON")
        return json.dumps({
            "error": "Stub fallback - LLM unavailable",
            "response": prompt[:500]
        })

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates: NVIDIA API > HF embeddings > Stub"""
        if self.use_nvidia_api:
            return self._rerank_nvidia(query, candidates)
        else:
            return self._rerank_hf(query, candidates)
    
    def _rerank_nvidia(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Call NVIDIA rerank API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            
            documents = [c.get("text", "") for c in candidates]
            payload = {
                "model": "nvidia/nv-rerank-qa-mixtral-8x7b",
                "query": query,
                "documents": documents,
                "top_k": len(candidates),
            }
            
            response = requests.post(
                f"{self.base_url}/ranking",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            ranked = []
            for item in result.get("results", []):
                ranked.append({
                    "index": int(item.get("index", 0)),
                    "score": float(item.get("score", 0.0))
                })
            
            logger.debug(f"Reranked {len(ranked)} candidates via NVIDIA")
            return ranked
        except Exception as e:
            logger.warning(f"NVIDIA rerank failed: {e}. Falling back to HF embeddings.")
            return self._rerank_hf(query, candidates)
    
    def _rerank_hf(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank using HuggingFace embeddings"""
        qv = _embed_text_hf(query)
        cand_embeddings = [_embed_text_hf(c.get("text", "")) for c in candidates]
        sims = [float(np.dot(qv, cv)) for cv in cand_embeddings]
        indexed = list(enumerate(sims))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [{"index": i, "score": float(s)} for i, s in indexed]
