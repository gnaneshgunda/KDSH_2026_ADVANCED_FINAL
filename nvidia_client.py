"""
NVIDIA NIM + HuggingFace Client for embeddings, chat, and reranking
Uses real NVIDIA API when available, falls back to HuggingFace Transformers

FIXED: Correct base URL (v2) to match endpoints (/openai/embeddings, /chat/completions)
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
    """Embed text using HuggingFace SentenceTransformer"""
    if HAS_SENTENCE_TRANSFORMERS:
        vec = _hf_embedder.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
    else:
        # Fallback: deterministic stub
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
        # FIXED: Use v2 API with openai-compatible endpoints
        self.base_url = base_url or "https://api.nvcf.nvidia.com/v2"
        self.use_nvidia_api = bool(api_key)
        self.use_hf = not self.use_nvidia_api and HAS_SENTENCE_TRANSFORMERS
        
        if self.use_nvidia_api:
            logger.info(f"Using NVIDIA API for inference: {self.base_url}")
        elif self.use_hf:
            logger.info("Using HuggingFace models for inference (embeddings)")
        else:
            logger.info("Using stub/deterministic fallback for inference")


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
                "Accept": "application/json",
            }
            payload = {"input": [t if t is not None else "" for t in texts]}
            
            response = requests.post(
                f"{self.base_url}/openai/embeddings",
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
            return [_embed_text_hf(t if t is not None else "") for t in texts]


    def chat(self, messages, temperature: float = 0.0) -> str:
        """Get chat response: NVIDIA API > Stub heuristic"""
        if self.use_nvidia_api:
            return self._chat_nvidia(messages, temperature)
        else:
            return self._chat_stub(messages, temperature)
    
    def _chat_nvidia(self, messages, temperature: float = 0.0) -> str:
        """Call NVIDIA chat API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            
            if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
                payload_messages = messages
            else:
                prompt = str(messages) if not isinstance(messages, list) else "\n".join(m.get("content", "") for m in messages)
                payload_messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": payload_messages,
                "temperature": temperature,
                "top_p": 1.0,
                "max_tokens": 1024,
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
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


        # If prompt expects JSON (heuristic: contains 'Return JSON') reply with a simple JSON
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


        # Default: return the prompt truncated for visibility
        return prompt[:1000]


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