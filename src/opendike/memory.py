from typing import Dict, List, Any, Optional
import numpy as np
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from src.opendike.models import MoralTrace
from src.opendike.config import config

class MemPalace:
    """
    Hierarchical memory structure (Wings -> Halls -> Rooms).
    Stores only morally relevant traces.
    """
    def __init__(self, embedding_model_name: str = None):
        if embedding_model_name is None:
            embedding_model_name = config.get("memory.embedding_model_name", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(embedding_model_name)
        # Structure: {layer_type: {layer_id: [MoralTrace]}}
        # e.g., {'country': {'US': [trace1, trace2]}, 'personal': {'user123': [...]}}
        self.storage: Dict[str, Dict[str, List[MoralTrace]]] = {
            "country": {},
            "community": {},
            "organization": {},
            "demographic": {},
            "personal": {}
        }

    def store_trace(self, layer_type: str, layer_id: str, trace: Dict[str, Any]):
        """Store a trace, ensuring it has content and embedding."""
        content = trace.get("content") or trace.get("query") or ""
        metadata = trace.get("metadata", {})
        timestamp = trace.get("timestamp")
        
        if isinstance(timestamp, datetime):
            ts = timestamp.timestamp()
        else:
            ts = timestamp or time.time()

        if layer_type not in self.storage:
            self.storage[layer_type] = {}
        if layer_id not in self.storage[layer_type]:
            self.storage[layer_type][layer_id] = []
            
        embedding = self.model.encode(content).tolist()
        mt = MoralTrace(
            content=content,
            embedding=embedding,
            metadata=metadata,
            timestamp=ts
        )
        self.storage[layer_type][layer_id].append(mt)

    def add_trace(self, layer_type: str, layer_id: str, content: str, metadata: Dict[str, Any] = None):
        """Legacy support for add_trace."""
        self.store_trace(layer_type, layer_id, {"content": content, "metadata": metadata})

    def retrieve_traces(self, layer_type: str, layer_id: str, query: str, top_k: int = None, min_relevance: float = 0.0) -> List[MoralTrace]:
        if top_k is None:
            top_k = config.get("memory.top_k", 3)
        if layer_type not in self.storage or layer_id not in self.storage[layer_type]:
            return []
            
        traces = self.storage[layer_type][layer_id]
        if not traces:
            return []
            
        query_embedding = self.model.encode(query)
        
        # Simple cosine similarity search
        results = []
        for t in traces:
            sim = np.dot(query_embedding, t.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(t.embedding))
            if sim >= min_relevance:
                results.append((sim, t))
            
        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        return [t for sim, t in results[:top_k]]
