from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import time
import json

# --- 1. Moral Vector Schema ---

class MoralVector(BaseModel):
    """
    Extended Moral Foundations Theory (MFT) + additional dimensions.
    Each value represents the weight/priority (0.0 - 1.0).
    """
    care_harm: float = Field(0.5, ge=0.0, le=1.0)
    fairness_proportionality: float = Field(0.5, ge=0.0, le=1.0)
    loyalty_betrayal: float = Field(0.5, ge=0.0, le=1.0)
    authority_subversion: float = Field(0.5, ge=0.0, le=1.0)
    sanctity_degradation: float = Field(0.5, ge=0.0, le=1.0)
    liberty_oppression: float = Field(0.5, ge=0.0, le=1.0)
    
    # Reasoning style hints
    deontological_vs_utilitarian: float = Field(0.5, ge=0.0, le=1.0, description="0: Utilitarian, 1: Deontological")
    
    # Metadata and Reasoning
    reasoning: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.care_harm, self.fairness_proportionality, self.loyalty_betrayal,
            self.authority_subversion, self.sanctity_degradation, self.liberty_oppression,
            self.deontological_vs_utilitarian
        ])

    @classmethod
    def from_numpy(cls, arr: np.ndarray, reasoning: str = None, constraints: List[str] = None):
        return cls(
            care_harm=float(arr[0]),
            fairness_proportionality=float(arr[1]),
            loyalty_betrayal=float(arr[2]),
            authority_subversion=float(arr[3]),
            sanctity_degradation=float(arr[4]),
            liberty_oppression=float(arr[5]),
            deontological_vs_utilitarian=float(arr[6]),
            reasoning=reasoning,
            constraints=constraints or []
        )

# --- 2. MemPalace Hierarchical Storage ---

class MoralTrace(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)

class MemPalace:
    """
    Hierarchical memory structure (Wings -> Halls -> Rooms).
    Stores only morally relevant traces.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
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

    def retrieve_traces(self, layer_type: str, layer_id: str, query: str, top_k: int = 3) -> List[MoralTrace]:
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

# --- 3. Layered Morality Deducer ---

class LayeredMoralityDeducer:
    """
    Composes moral vectors from different layers.
    Includes conflict detection and attention-style fusion.
    """
    def __init__(self, mem_palace: MemPalace):
        self.mem_palace = mem_palace
        # Default profiles for initialization (synthetic)
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

    def get_layer_vector(self, layer_type: str, layer_id: str, query: str) -> MoralVector:
        # 1. Start with base profile if exists
        base_v = self.base_profiles.get(layer_type, {}).get(layer_id, MoralVector())
        
        # 2. Retrieve traces to modulate
        traces = self.mem_palace.retrieve_traces(layer_type, layer_id, query)
        if not traces:
            return base_v
            
        # 3. Simple modulation (In a real system, this would be a small model update)
        # Here we just average the base with 'influence' from traces if they have vector updates
        # For prototype, we simulate trace influence
        vec = base_v.to_numpy()
        for t in traces:
            if "vector_delta" in t.metadata:
                delta = np.array(t.metadata["vector_delta"])
                vec = np.clip(vec + delta, 0, 1)
        
        return MoralVector.from_numpy(vec, reasoning=f"Based on {layer_id} {layer_type} priors and {len(traces)} retrieved traces.")

    def deduce(self, query: str, context: Dict[str, str]) -> Dict[str, Any]:
        """
        context example: {"country": "Nordic", "demographic": "teenager", "user_id": "u123"}
        """
        layers = {}
        # Fetch vectors for each provided context layer
        for layer_type, layer_id in context.items():
            if layer_type == "user_id":
                layers["personal"] = self.get_layer_vector("personal", layer_id, query)
            else:
                layers[layer_type] = self.get_layer_vector(layer_type, layer_id, query)
                
        # Composition: Weighted sum with hierarchy priority
        # Priority: personal (0.5) > demographic (0.2) > community (0.2) > country (0.1)
        weights = {"personal": 0.5, "demographic": 0.2, "community": 0.2, "country": 0.1}
        
        # Normalize weights based on present layers
        active_weights = {k: weights.get(k, 0.1) for k in layers.keys()}
        total_w = sum(active_weights.values())
        for k in active_weights: active_weights[k] /= total_w
        
        composite_vec = np.zeros(7)
        reasoning_parts = []
        all_constraints = []
        
        for name, vec in layers.items():
            composite_vec += vec.to_numpy() * active_weights[name]
            reasoning_parts.append(f"{name.capitalize()} ({active_weights[name]:.2f}): {vec.reasoning}")
            all_constraints.extend(vec.constraints)
            
        # Conflict detection
        conflicts = []
        # Example: if one layer is high authority and another is low
        auth_values = [vec.authority_subversion for vec in layers.values()]
        if max(auth_values) - min(auth_values) > 0.4:
            conflicts.append("High variance in Authority priority between layers.")
            
        liberty_values = [vec.liberty_oppression for vec in layers.values()]
        if max(liberty_values) - min(liberty_values) > 0.4:
            conflicts.append("High variance in Liberty priority between layers.")

        return {
            "composite_vector": MoralVector.from_numpy(composite_vec),
            "layer_contributions": layers,
            "conflicts": conflicts,
            "reasoning": " | ".join(reasoning_parts),
            "constraints": list(set(all_constraints))
        }

# --- 4. Morality Wrapper ---

class MoralityWrapper:
    """
    Gateway that wraps any downstream LLM.
    """
    def __init__(self, deducer: LayeredMoralityDeducer, safety_floor: List[str] = None):
        self.deducer = deducer
        self.safety_floor = safety_floor or [
            "Do not encourage self-harm.",
            "Do not generate hate speech or direct incitement to violence."
        ]

    def _generate_system_prefix(self, deduction: Dict[str, Any]) -> str:
        mv: MoralVector = deduction["composite_vector"]
        prefix = f"SYSTEM MORAL STEERING:\n"
        prefix += f"Prioritize: Care ({mv.care_harm:.2f}), Fairness ({mv.fairness_proportionality:.2f}), "
        prefix += f"Loyalty ({mv.loyalty_betrayal:.2f}), Authority ({mv.authority_subversion:.2f}), "
        prefix += f"Sanctity ({mv.sanctity_degradation:.2f}), Liberty ({mv.liberty_oppression:.2f}).\n"
        
        style = "utilitarian" if mv.deontological_vs_utilitarian < 0.4 else "deontological" if mv.deontological_vs_utilitarian > 0.6 else "balanced"
        prefix += f"Reasoning Style: {style}.\n"
        
        if deduction["constraints"]:
            prefix += f"Constraints: {', '.join(deduction['constraints'])}\n"
            
        prefix += "Safety Floor: " + "; ".join(self.safety_floor) + "\n"
        prefix += f"Transparency: {deduction['reasoning']}\n"
        
        if deduction["conflicts"]:
            prefix += f"Note: Moral conflicts detected ({', '.join(deduction['conflicts'])}). Seek a pluralistic synthesis.\n"
            
        return prefix

    def call_llm(self, llm_func, query: str, context: Dict[str, str]):
        start_time = time.time()
        
        # 1. Deduce moral vector
        deduction = self.deducer.deduce(query, context)
        
        # 2. Generate prefix
        prefix = self._generate_system_prefix(deduction)
        
        # 3. Call downstream
        # In a real app, you'd prepend prefix to system prompt or use as steering
        full_prompt = f"{prefix}\n\nUser Query: {query}"
        
        latency = (time.time() - start_time) * 1000
        print(f"[MoralityWrapper] Deducer Latency: {latency:.2f}ms")
        
        # Simulating LLM call
        return llm_func(full_prompt)

# --- 5. Continual Learning Loop ---

class ContinualLearner:
    def __init__(self, mem_palace: MemPalace):
        self.mem_palace = mem_palace

    def extract_salient_episodes(self, query: str, response: str, feedback: str):
        # This would be a small LLM call or heuristic to check if interaction was morally salient
        if any(keyword in feedback.lower() for keyword in ["moral", "wrong", "right", "better", "should"]):
            # Heuristic: extract a 'vector delta' based on feedback
            delta = np.zeros(7)
            if "authority" in feedback.lower() or "respect" in feedback.lower():
                delta[3] = 0.15 if "more" in feedback.lower() or "respect" in feedback.lower() else -0.15
            if "care" in feedback.lower() or "harm" in feedback.lower():
                delta[0] = 0.15 if "better" in feedback.lower() else -0.15
            if "liberty" in feedback.lower() or "freedom" in feedback.lower():
                delta[5] = 0.15 if "more" in feedback.lower() else -0.15
                
            return {
                "content": f"Query: {query} | Response: {response} | Feedback: {feedback}",
                "vector_delta": delta.tolist()
            }
        return None

    def update_personal_layer(self, user_id: str, episode: Dict[str, Any]):
        self.mem_palace.add_trace(
            layer_type="personal",
            layer_id=user_id,
            content=episode["content"],
            metadata={"vector_delta": episode["vector_delta"]}
        )

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Initialize components
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    wrapper = MoralityWrapper(deducer)
    learner = ContinualLearner(palace)
    
    # 2. Mock LLM function
    def mock_llm(prompt):
        return f"Mock Response to: {prompt[:100]}..."

    # 3. Scenario: Nordic Teenager
    context = {"country": "Nordic", "demographic": "teenager", "user_id": "user_456"}
    query = "Should I follow my parents' rules even if I disagree?"
    
    print("--- Initial Call ---")
    response = wrapper.call_llm(mock_llm, query, context)
    print(response)
    
    # 4. Continual Learning: User gives feedback
    feedback = "I liked that you prioritized my liberty, but you should respect authority more."
    episode = learner.extract_salient_episodes(query, response, feedback)
    if episode:
        learner.update_personal_layer("user_456", episode)
        print("\n[Learning] Personal layer updated with moral trace.")
    
    # 5. Subsequent call (should be influenced by trace)
    print("\n--- Subsequent Call (Should show increased Authority) ---")
    response_2 = wrapper.call_llm(mock_llm, query, context)
    print(response_2)

    # 6. Scenario: Cultural Conflict
    print("\n--- Conflict Scenario (Nordic vs Traditional) ---")
    conflict_context = {
        "country": "Nordic",
        "community": "RuralTraditional",
        "user_id": "user_789"
    }
    conflict_query = "Is it okay to publicly challenge a community elder?"
    deduction = deducer.deduce(conflict_query, conflict_context)
    print(f"Conflicts detected: {deduction['conflicts']}")
    print(wrapper.call_llm(mock_llm, conflict_query, conflict_context))
