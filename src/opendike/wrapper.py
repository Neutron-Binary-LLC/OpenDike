from typing import Dict, List, Any
import time
from opendike.models import MoralVector
from opendike.experts import LayeredMoralityDeducer
from opendike.config import config

class MoralityWrapper:
    """
    Gateway that wraps any downstream LLM.
    """
    def __init__(self, deducer: LayeredMoralityDeducer, safety_floor: List[str] = None):
        self.deducer = deducer
        if safety_floor is None:
            safety_floor = config.get("wrapper.safety_floor", [
                "Do not encourage self-harm.",
                "Do not generate hate speech or direct incitement to violence."
            ])
        self.safety_floor = safety_floor

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
        deduction = self.deducer.deduce_moral_vector(query, context)
        
        # 2. Generate prefix
        prefix = self._generate_system_prefix(deduction)
        
        # 3. Call downstream
        # In a real app, you'd prepend prefix to system prompt or use as steering
        full_prompt = f"{prefix}\n\nUser Query: {query}"
        
        latency = (time.time() - start_time) * 1000
        print(f"[MoralityWrapper] Deducer Latency: {latency:.2f}ms")
        
        # Simulating LLM call
        return llm_func(full_prompt)
