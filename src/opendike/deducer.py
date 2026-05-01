from typing import Dict, List, Any
import numpy as np
from src.opendike.models import MoralVector
from src.opendike.memory import MemPalace
from src.opendike.experts import MoralExpert, MoralGatingNetwork
from src.opendike.config import config

class LayeredMoralityDeducer:
    """
    Acts as the MoE Controller.
    Composes moral vectors using Experts and a Gating Network.
    """
    def __init__(self, mem_palace: MemPalace):
        self.mem_palace = mem_palace
        self.gating_network = MoralGatingNetwork()
        # Default profiles for initialization (synthetic)
        raw_profiles = config.get("deducer.base_profiles", {})
        self.base_profiles = {}
        for l_type, profiles in raw_profiles.items():
            self.base_profiles[l_type] = {
                l_id: MoralVector(**p_data) for l_id, p_data in profiles.items()
            }
        
        # Fallback if config is missing
        if not self.base_profiles:
            self.base_profiles = {
                "country": {
                    "Nordic": MoralVector(care_harm=0.9, fairness_proportionality=0.9, liberty_oppression=0.8, authority_subversion=0.3, reasoning="Prioritizes social welfare, equality, and individual autonomy."),
                    "EastAsian": MoralVector(authority_subversion=0.8, loyalty_betrayal=0.8, sanctity_degradation=0.7, care_harm=0.6, reasoning="Emphasizes social harmony, filial piety, and respect for hierarchy."),
                    "US": MoralVector(liberty_oppression=0.9, fairness_proportionality=0.8, care_harm=0.7, reasoning="Strong emphasis on individual liberty and market-based fairness.")
                },
                "community": {
                    "UrbanProgressive": MoralVector(care_harm=0.9, liberty_oppression=0.9, authority_subversion=0.2, reasoning="Focus on inclusivity and social justice."),
                    "RuralTraditional": MoralVector(sanctity_degradation=0.8, authority_subversion=0.8, loyalty_betrayal=0.8, reasoning="Values tradition, community cohesion, and established order.")
                },
                "demographic": {
                    "teenager": MoralVector(liberty_oppression=0.9, care_harm=0.7, authority_subversion=0.2, reasoning="High drive for autonomy and peer loyalty."),
                    "elder": MoralVector(authority_subversion=0.8, sanctity_degradation=0.7, loyalty_betrayal=0.7, reasoning="Respect for experience and preservation of cultural heritage.")
                }
            }

    def _get_expert(self, layer_type: str, layer_id: str) -> MoralExpert:
        base_v = self.base_profiles.get(layer_type, {}).get(layer_id, MoralVector())
        return MoralExpert(layer_type, layer_id, base_v, self.mem_palace)

    def deduce(self, query: str, context: Dict[str, str]) -> Dict[str, Any]:
        """
        MoE-style deduction.
        """
        experts = {}
        for layer_type, layer_id in context.items():
            l_type = "personal" if layer_type == "user_id" else layer_type
            experts[l_type] = self._get_expert(l_type, layer_id)
                
        # Gating: Get routing weights
        routing_weights = self.gating_network.route(query, list(experts.keys()))
        
        # Expert Inference
        expert_outputs = {name: expert.get_vector(query) for name, expert in experts.items()}
        
        # Fusion
        composite_vec = np.zeros(7)
        reasoning_parts = []
        all_constraints = []
        
        for name, vec in expert_outputs.items():
            weight = routing_weights[name]
            composite_vec += vec.to_numpy() * weight
            reasoning_parts.append(f"Expert-{name} (weight {weight:.2f}): {vec.reasoning}")
            all_constraints.extend(vec.constraints)
            
        # Conflict detection
        conflicts = []
        expert_vecs = [v.to_numpy() for v in expert_outputs.values()]
        if len(expert_vecs) > 1:
            auth_values = [v[3] for v in expert_vecs]
            if max(auth_values) - min(auth_values) > 0.4:
                conflicts.append("High variance in Authority priority between experts.")
            
            liberty_values = [v[5] for v in expert_vecs]
            if max(liberty_values) - min(liberty_values) > 0.4:
                conflicts.append("High variance in Liberty priority between experts.")

        return {
            "composite_vector": MoralVector.from_numpy(composite_vec),
            "expert_outputs": expert_outputs,
            "routing_weights": routing_weights,
            "conflicts": conflicts,
            "reasoning": " | ".join(reasoning_parts),
            "constraints": list(set(all_constraints))
        }
