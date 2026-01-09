"""
NVIDIA NIM API Client for embeddings and LLM completions
"""

import logging
from typing import List, Dict
import requests
import numpy as np
from config import NVIDIA_BASE_URL, EMBEDDING_MODEL, CHAT_MODEL

logger = logging.getLogger(__name__)


class NVIDIAClient:
    """Client for NVIDIA NIM API"""

    def __init__(self, api_key: str, base_url: str = NVIDIA_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = EMBEDDING_MODEL
        self.chat_model = CHAT_MODEL
        logger.info(f"NVIDIA client initialized with base URL: {base_url}")

    def _get_headers(self) -> Dict[str, str]:
        """Generate appropriate headers based on API endpoint type"""
        headers = {"Content-Type": "application/json"}
        
        if "api.nvidia.com" in self.base_url:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            if self.api_key != "local":
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using NVIDIA NIM
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        url = f"{self.base_url}/embeddings"
        headers = self._get_headers()

        payload = {
            "model": self.embedding_model,
            "input": texts
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            embeddings = np.array([item["embedding"] for item in data["data"]])
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    def chat(self, messages: List[Dict], temperature: float = 0.0) -> str:
        """
        Get chat/completion response from NVIDIA LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Response text from model
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 400
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.debug(f"Chat response received ({len(content)} chars)")
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise
