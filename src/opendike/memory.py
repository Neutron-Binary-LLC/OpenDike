from typing import Dict, List, Any, Optional
import numpy as np
import time
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from opendike.models import MoralTrace
from opendike.config import config

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
        self.storage_path = config.get("memory.storage_path", "data/memory_storage.json")
        self.load_storage()

    def save_storage(self):
        """Persist storage to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        serializable_storage = {}
        for l_type, l_ids in self.storage.items():
            serializable_storage[l_type] = {}
            for l_id, traces in l_ids.items():
                serializable_storage[l_type][l_id] = [t.dict() for t in traces]
        
        import json
        with open(self.storage_path, 'w') as f:
            json.dump(serializable_storage, f, indent=2)

    def load_storage(self):
        """Load storage from disk."""
        if not os.path.exists(self.storage_path):
            return

        import json
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for l_type, l_ids in data.items():
                if l_type not in self.storage:
                    self.storage[l_type] = {}
                for l_id, traces_data in l_ids.items():
                    self.storage[l_type][l_id] = [MoralTrace(**t) for t in traces_data]
        except Exception as e:
            print(f"Error loading memory storage: {e}")

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
