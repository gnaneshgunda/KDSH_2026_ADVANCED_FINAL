"""
LangChain NVIDIA client with API key rotation
"""

import numpy as np
from typing import List
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from api_key_manager import get_next_api_key


class LangChainNVIDIAClient:
    def __init__(self, embedding_config: dict, chat_config: dict):
        self.emb_cfg = embedding_config
        self.chat_cfg = chat_config
        self._init_clients()
    
    def _init_clients(self):
        key = get_next_api_key()
        self.embedding_client = NVIDIAEmbeddings(**{**self.emb_cfg, "api_key": key})
        self.chat_client = ChatNVIDIA(**{**self.chat_cfg, "api_key": key})
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.embedding_client.embed_documents(texts)
        self._init_clients()
        return [np.array(e, dtype=np.float32) for e in embeddings]
    
    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.embedding_client.embed_query(text)
        self._init_clients()
        return np.array(embedding, dtype=np.float32)
    
    def chat_completion(self, messages: List[dict]) -> str:
        response = self.chat_client.invoke(messages)
        self._init_clients()
        return response.content
    
    def chat_stream(self, messages: List[dict]):
        for chunk in self.chat_client.stream(messages):
            if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                yield chunk.additional_kwargs["reasoning_content"]
            yield chunk.content
        self._init_clients()
