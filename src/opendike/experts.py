from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import json
from datetime import datetime
import logging

from opendike.models import MoralVector
from opendike.memory import MemPalace
from opendike.config import config

logger = logging.getLogger(__name__)


@dataclass
class MoralExpert:
    """Specialized moral expert for a specific hierarchical layer."""

    layer_type: str  # e.g., "country", "community", "demographic", "personal"
    layer_id: str  # e.g., "US", "Muslim_Conservative", "teenager_18_24", "user_123"
    base_profile: MoralVector
    mem_palace: MemPalace

    # Lightweight adapter for continual learning (per-expert)
    adapter: Optional[Dict] = field(default_factory=dict)  # e.g., LoRA-style or simple bias

    def get_vector(self, query: str, context: Optional[Dict] = None) -> MoralVector:
        """Retrieve modulated moral vector for this layer."""
        try:
            traces = self.mem_palace.retrieve_traces(
                layer_type=self.layer_type,
                layer_id=self.layer_id,
                query=query,
                top_k=8,  # configurable
                min_relevance=0.65
            )

            vec = self.base_profile.to_numpy().copy()

            if traces:
                for trace in traces:
                    if "vector_delta" in trace.metadata:
                        delta = np.array(trace.metadata["vector_delta"], dtype=float)
                        # Apply learning rate decay based on trace age
                        age_factor = self._get_age_factor(trace.timestamp)
                        vec = np.clip(vec + (delta * age_factor), 0.0, 1.0)

            # Optional: Apply expert-specific adapter (for continual fine-tuning)
            if self.adapter and "bias" in self.adapter:
                vec = np.clip(vec + np.array(self.adapter["bias"]), 0.0, 1.0)

            return MoralVector.from_numpy(
                vec,
                reasoning=f"Layer: {self.layer_type}/{self.layer_id} | "
                          f"Traces used: {len(traces)} | Updated: {datetime.now().isoformat()}",
                source_layer=self.layer_type,
                layer_id=self.layer_id
            )

        except Exception as e:
            logger.error(f"Error in MoralExpert.get_vector for {self.layer_type}/{self.layer_id}: {e}")
            return self.base_profile  # safe fallback

    def _get_age_factor(self, timestamp: Optional[datetime]) -> float:
        """Decay influence of old traces."""
        if not timestamp:
            return 0.8
        days_old = (datetime.now() - timestamp).days
        return max(0.3, 1.0 - (days_old * 0.015))  # gentle decay


class MoralGatingNetwork:
    """Hierarchical-aware gating network for combining expert vectors."""

    def __init__(self, default_weights: Optional[Dict] = None):
        self.default_weights = default_weights or config.get(
            "experts.default_weights",
            {
                "personal": 0.45,
                "demographic": 0.20,
                "community": 0.15,
                "organization": 0.10,
                "country": 0.10
            }
        )
        self.intent_config = config.get("experts.intent_gating", {
            "mode": "layered",
            "intents": {}
        })

    def route(self, query: str, active_layers: List[str],
              user_context: Optional[Dict] = None) -> Dict[str, float]:
        """Compute routing weights with hierarchy bias and query intent detection."""
        weights = {}
        query_lower = query.lower()
        mode = self.intent_config.get("mode", "layered")
        intents = self.intent_config.get("intents", {})

        # 1. Base weights and specificity bonus
        for layer in active_layers:
            base_w = self.default_weights.get(layer, 0.1)
            # Hierarchy bonus: more specific layers get slight boost
            specificity_bonus = 0.08 if layer == "personal" else 0.0
            weights[layer] = base_w + specificity_bonus

        # 2. Intent-based dynamic boost
        if mode == "layered":
            # Current behavior: boost specific layers if intent matches
            for intent_name, data in intents.items():
                keywords = data.get("keywords", [])
                if any(word in query_lower for word in keywords):
                    boost = data.get("boost", 0.15)
                    target_layers = data.get("target_layers", [])
                    for layer in target_layers:
                        if layer in weights:
                            weights[layer] += boost
        
        elif mode == "expert":
            # Aggressive behavior: redistribute weights towards the primary intent
            primary_intent = None
            max_hits = 0
            for intent_name, data in intents.items():
                hits = sum(1 for word in data.get("keywords", []) if word in query_lower)
                if hits > max_hits:
                    max_hits = hits
                    primary_intent = data
            
            if primary_intent:
                boost = primary_intent.get("boost", 0.3)
                target_layers = primary_intent.get("target_layers", [])
                for layer in target_layers:
                    if layer in weights:
                        # In expert mode, we boost target layers and potentially suppress others
                        weights[layer] += boost
                
                # Suppress non-target, non-personal layers slightly to sharpen focus
                for layer in weights:
                    if layer not in target_layers and layer != "personal":
                        weights[layer] = max(0.01, weights[layer] - (boost / 2))

        # 3. Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        else:
            weights = {k: round(1.0 / len(active_layers), 4) for k in active_layers}

        return weights


class LayeredMoralityDeducer:
    """Main orchestrator: Layered Morality Expert System."""

    def __init__(self, mem_palace: MemPalace):
        self.mem_palace = mem_palace
        self.experts: Dict[str, MoralExpert] = {}
        self.gating = MoralGatingNetwork()

        # Load base profiles from config / disk
        self._initialize_base_experts()

    def _initialize_base_experts(self):
        """Load or create base moral profiles per layer."""
        # Check both "moral.base_profiles" and "deducer.base_profiles" for backward compatibility
        base_configs = config.get("moral.base_profiles") or config.get("deducer.base_profiles", {})
        
        # Fallback to hardcoded profiles if config is missing
        if not base_configs:
            base_configs = {
                "organization": {
                    "TechCorp": {"care_harm": 0.8, "loyalty_betrayal": 0.9, "authority_subversion": 0.8, "reasoning": "Efficiency and loyalty."},
                    "NonProfit": {"care_harm": 0.95, "fairness_proportionality": 0.9, "reasoning": "Altruism and impact."}
                },
                "country": {
                    "Nordic": {"care_harm": 0.9, "fairness_proportionality": 0.9, "liberty_oppression": 0.8, "authority_subversion": 0.3, "reasoning": "Prioritizes social welfare, equality, and individual autonomy."},
                    "EastAsian": {"authority_subversion": 0.8, "loyalty_betrayal": 0.8, "sanctity_degradation": 0.7, "care_harm": 0.6, "reasoning": "Emphasizes social harmony, filial piety, and respect for hierarchy."},
                    "US": {"liberty_oppression": 0.9, "fairness_proportionality": 0.8, "care_harm": 0.7, "reasoning": "Strong emphasis on individual liberty and market-based fairness."}
                },
                "community": {
                    "UrbanProgressive": {"care_harm": 0.9, "liberty_oppression": 0.9, "authority_subversion": 0.2, "reasoning": "Focus on inclusivity and social justice."},
                    "RuralTraditional": {"sanctity_degradation": 0.8, "authority_subversion": 0.8, "loyalty_betrayal": 0.8, "reasoning": "Values tradition, community cohesion, and established order."}
                },
                "demographic": {
                    "teenager": {"liberty_oppression": 0.9, "care_harm": 0.7, "authority_subversion": 0.2, "reasoning": "High drive for autonomy and peer loyalty."},
                    "elder": {"authority_subversion": 0.8, "sanctity_degradation": 0.7, "loyalty_betrayal": 0.7, "reasoning": "Respect for experience and preservation of cultural heritage."}
                }
            }

        for layer_type, profiles in base_configs.items():
            for layer_id, profile_dict in profiles.items():
                mv = MoralVector.from_dict(profile_dict)
                self.register_expert(layer_type, layer_id, mv)
            
            # Register a 'default' for each layer type if not present
            if 'default' not in profiles:
                first_profile = next(iter(profiles.values()))
                self.register_expert(layer_type, 'default', MoralVector.from_dict(first_profile))

    def register_expert(self, layer_type: str, layer_id: str, base_profile: MoralVector):
        """Register a new moral expert."""
        key = f"{layer_type}:{layer_id}"
        self.experts[key] = MoralExpert(
            layer_type=layer_type,
            layer_id=layer_id,
            base_profile=base_profile,
            mem_palace=self.mem_palace
        )

    def deduce_moral_vector(self,
                            query: str,
                            context: Dict[str, str]) -> Dict[str, Any]:
        """
        Main entry point: Produce composite steering vector and detailed analysis.
        Replaces the old 'deduce' and 'deduce_moral_vector' methods.
        """
        active_layers = []
        user_context = {}
        
        # Map input context to internal layers
        for key, value in context.items():
            if key == "user_id":
                layer_type = "personal"
            elif key == "org_id":
                layer_type = "organization"
            else:
                layer_type = key
            
            active_layers.append(layer_type)
            user_context[layer_type] = value

        if not active_layers:
            active_layers = ["country"]
            user_context = {"country": "default"}

        # Get routing weights
        weights = self.gating.route(query, active_layers, user_context)

        # Get vectors from active experts
        expert_outputs: Dict[str, MoralVector] = {}
        expert_vectors: List[MoralVector] = []
        expert_weights: List[float] = []

        for layer in active_layers:
            layer_id = user_context.get(layer, "default")
            key = f"{layer}:{layer_id}"
            expert = self.experts.get(key) or self._get_fallback_expert(layer)

            if expert:
                vec = expert.get_vector(query, user_context)
                expert_outputs[layer] = vec
                expert_vectors.append(vec)
                expert_weights.append(weights.get(layer, 0.1))

        # Composite via weighted average
        if not expert_vectors:
            composite = MoralVector.default()
        else:
            composite = MoralVector.weighted_average(expert_vectors, expert_weights)

        # Conflict detection (from deducer.py)
        conflicts = []
        expert_vecs = [v.to_numpy() for v in expert_outputs.values()]
        if len(expert_vecs) > 1:
            # Dimension-wise conflict detection
            dimensions = [
                "Care/Harm", "Fairness", "Loyalty", 
                "Authority", "Sanctity", "Liberty", "Style"
            ]
            for i, dim_name in enumerate(dimensions):
                values = [v[i] for v in expert_vecs]
                if max(values) - min(values) > 0.4:
                    conflicts.append(f"High variance in {dim_name} priority between experts.")

        # Transparency and Metadata
        composite.metadata = composite.metadata or {}
        composite.metadata.update({
            "layers_used": active_layers,
            "routing_weights": weights,
            "timestamp": datetime.now().isoformat(),
            "conflicts": conflicts
        })

        return {
            "composite_vector": composite,
            "expert_outputs": expert_outputs,
            "routing_weights": weights,
            "conflicts": conflicts,
            "reasoning": composite.reasoning,
            "constraints": composite.constraints
        }

    def deduce(self, query: str, context: Dict[str, str]) -> Dict[str, Any]:
        """Alias for compatibility."""
        return self.deduce_moral_vector(query, context)

    def _get_fallback_expert(self, layer_type: str) -> Optional[MoralExpert]:
        """Return a reasonable fallback."""
        key = f"{layer_type}:default"
        return self.experts.get(key)

    def update_from_feedback(self,
                             query: str,
                             final_vector: MoralVector,
                             user_feedback: Dict[str, Any],
                             active_layers: List[str]):
        """Continual learning: Update relevant experts from feedback."""
        for layer in active_layers:
            # Determine layer_id (e.g., user_id for personal, country_id for country)
            # Default to 'default' if not provided in feedback
            layer_id = user_feedback.get('layer_id', 'default')
            
            # Map feedback layer_id to expert key
            key = f"{layer}:{layer_id}"
            expert = self.experts.get(key)
            
            # If expert doesn't exist for a personal layer, create it
            if not expert and layer == "personal":
                self.register_expert(layer, layer_id, MoralVector.default())
                expert = self.experts.get(key)
                
            if expert:
                # Create delta from feedback
                delta = self._compute_feedback_delta(final_vector, user_feedback)
                trace = {
                    "query": query,
                    "timestamp": datetime.now(),
                    "metadata": {
                        "vector_delta": delta.tolist(),
                        "feedback_score": user_feedback.get("satisfaction", 0.5),
                        "source": "user_feedback"
                    }
                }
                expert.mem_palace.store_trace(
                    layer_type=expert.layer_type,
                    layer_id=expert.layer_id,
                    trace=trace
                )

                # Optional: lightweight adapter update
                if not hasattr(expert, 'adapter') or expert.adapter is None:
                    expert.adapter = {}
                # Simple moving average bias update
                if "bias" not in expert.adapter:
                    expert.adapter["bias"] = (delta * 0.1).tolist()
                else:
                    expert.adapter["bias"] = (np.array(expert.adapter["bias"]) * 0.7 +
                                              np.array(delta) * 0.3).tolist()

    def _compute_feedback_delta(self, final_vector: MoralVector, user_feedback: Dict[str, Any]) -> np.ndarray:
        """Heuristic to compute vector delta from feedback."""
        delta = np.zeros(7)
        feedback_text = user_feedback.get("text", "").lower()
        step = config.get("learning.vector_delta_step", 0.15)
        
        if "authority" in feedback_text or "respect" in feedback_text:
            delta[3] = step if "more" in feedback_text or "respect" in feedback_text else -step
        if "care" in feedback_text or "harm" in feedback_text:
            delta[0] = step if "better" in feedback_text else -step
        if "liberty" in feedback_text or "freedom" in feedback_text:
            delta[5] = step if "more" in feedback_text else -step
            
        return delta


# ================ Helper / Utility ================

def create_steering_prompt(moral_vector: MoralVector) -> str:
    """Convert moral vector to a clean system prompt for downstream LLM."""
    weights = moral_vector.to_dict()["weights"]
    prompt = f"""You are responding with the following moral priorities (0-1 scale):
Care/Harm: {weights.get('care', 0.5):.2f}
Fairness: {weights.get('fairness', 0.5):.2f}
Loyalty: {weights.get('loyalty', 0.5):.2f}
Authority: {weights.get('authority', 0.5):.2f}
Sanctity/Purity: {weights.get('sanctity', 0.5):.2f}
Liberty: {weights.get('liberty', 0.5):.2f}

Respect the user's cultural, age, and personal context as reflected in these weights.
Prioritize nuance and avoid moralizing from a single cultural viewpoint.
"""
    return prompt.strip()