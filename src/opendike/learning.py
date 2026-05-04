from typing import Dict, Any
import numpy as np
from opendike.memory import MemPalace
from opendike.config import config

class ContinualLearner:
    def __init__(self, mem_palace: MemPalace):
        self.mem_palace = mem_palace

    def extract_salient_episodes(self, query: str, response: str, feedback: str):
        # This would be a small LLM call or heuristic to check if interaction was morally salient
        keywords = config.get("learning.salience_keywords", ["moral", "wrong", "right", "better", "should"])
        if any(keyword in feedback.lower() for keyword in keywords):
            # Heuristic: extract a 'vector delta' based on feedback
            delta = np.zeros(7)
            step = config.get("learning.vector_delta_step", 0.15)
            if "authority" in feedback.lower() or "respect" in feedback.lower():
                delta[3] = step if "more" in feedback.lower() or "respect" in feedback.lower() else -step
            if "care" in feedback.lower() or "harm" in feedback.lower():
                delta[0] = step if "better" in feedback.lower() else -step
            if "liberty" in feedback.lower() or "freedom" in feedback.lower():
                delta[5] = step if "more" in feedback.lower() else -step
                
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
