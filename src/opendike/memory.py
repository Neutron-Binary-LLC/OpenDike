from typing import Dict, List, Any
import numpy as np
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
            "demographic": {},
            "personal": {}
        }

    def add_trace(self, layer_type: str, layer_id: str, content: str, metadata: Dict[str, Any] = None):
        if layer_type not in self.storage:
            self.storage[layer_type] = {}
        if layer_id not in self.storage[layer_type]:
            self.storage[layer_type][layer_id] = []
            
        embedding = self.model.encode(content).tolist()
        trace = MoralTrace(
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.storage[layer_type][layer_id].append(trace)

    def retrieve_traces(self, layer_type: str, layer_id: str, query: str, top_k: int = None) -> List[MoralTrace]:
        if top_k is None:
            top_k = config.get("memory.top_k", 3)
        if layer_type not in self.storage or layer_id not in self.storage[layer_type]:
            return []
            
        traces = self.storage[layer_type][layer_id]
        if not traces:
            return []
            
        query_embedding = self.model.encode(query)
        
        # Simple cosine similarity search
        similarities = []
        for t in traces:
            sim = np.dot(query_embedding, t.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(t.embedding))
            similarities.append(sim)
            
        sorted_indices = np.argsort(similarities)[::-1]
        return [traces[i] for i in sorted_indices[:top_k]]
