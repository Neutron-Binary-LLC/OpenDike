from typing import Dict, List
import numpy as np
from src.opendike.models import MoralVector
from src.opendike.memory import MemPalace
from src.opendike.config import config

class MoralExpert:
    """
    Base class for a Moral Expert in the MoE setting.
    Each expert specializes in a specific layer (e.g., Country, Personal).
    """
    def __init__(self, layer_type: str, layer_id: str, base_profile: MoralVector, mem_palace: MemPalace):
        self.layer_type = layer_type
        self.layer_id = layer_id
        self.base_profile = base_profile
        self.mem_palace = mem_palace

    def get_vector(self, query: str) -> MoralVector:
        # Retrieve traces to modulate
        traces = self.mem_palace.retrieve_traces(self.layer_type, self.layer_id, query)
        if not traces:
            return self.base_profile
            
        vec = self.base_profile.to_numpy()
        for t in traces:
            if "vector_delta" in t.metadata:
                delta = np.array(t.metadata["vector_delta"])
                vec = np.clip(vec + delta, 0, 1)
        
        return MoralVector.from_numpy(
            vec, 
            reasoning=f"Expert for {self.layer_id} ({self.layer_type}) based on priors and {len(traces)} traces."
        )

class MoralGatingNetwork:
    """
    Learns/Calculates routing weights for the Mixture of Experts.
    In a training setting, this would be a learnable neural network.
    """
    def __init__(self, default_weights: Dict[str, float] = None):
        if default_weights is None:
            default_weights = config.get("experts.default_weights", {
                "personal": 0.4,
                "organization": 0.2,
                "demographic": 0.15,
                "community": 0.15,
                "country": 0.1
            })
        self.weights = default_weights

    def route(self, query: str, active_layers: List[str]) -> Dict[str, float]:
        # Simple heuristic routing for prototype.
        # In full MoE, this would take query embeddings as input.
        active_weights = {k: self.weights.get(k, 0.1) for k in active_layers}
        total_w = sum(active_weights.values())
        if total_w == 0: return {k: 1.0/len(active_layers) for k in active_layers}
        return {k: v / total_w for k, v in active_weights.items()}
