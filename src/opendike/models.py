from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import numpy as np
import time

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
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.care_harm, self.fairness_proportionality, self.loyalty_betrayal,
            self.authority_subversion, self.sanctity_degradation, self.liberty_oppression,
            self.deontological_vs_utilitarian
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                "care": self.care_harm,
                "fairness": self.fairness_proportionality,
                "loyalty": self.loyalty_betrayal,
                "authority": self.authority_subversion,
                "sanctity": self.sanctity_degradation,
                "liberty": self.liberty_oppression
            },
            "reasoning": self.reasoning,
            "constraints": self.constraints,
            "metadata": self.metadata
        }

    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # Handle both flat and nested 'weights' structure
        if "weights" in data:
            weights = data["weights"]
            return cls(
                care_harm=weights.get("care", 0.5),
                fairness_proportionality=weights.get("fairness", 0.5),
                loyalty_betrayal=weights.get("loyalty", 0.5),
                authority_subversion=weights.get("authority", 0.5),
                sanctity_degradation=weights.get("sanctity", 0.5),
                liberty_oppression=weights.get("liberty", 0.5),
                deontological_vs_utilitarian=data.get("deontological_vs_utilitarian", 0.5),
                reasoning=data.get("reasoning"),
                constraints=data.get("constraints", [])
            )
        return cls(**data)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, reasoning: str = None, constraints: List[str] = None, source_layer: str = None, layer_id: str = None):
        mv = cls(
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
        if source_layer or layer_id:
            mv.metadata = {"source_layer": source_layer, "layer_id": layer_id}
        return mv

    @staticmethod
    def weighted_average(vectors: List['MoralVector'], weights: List[float]) -> 'MoralVector':
        if not vectors:
            return MoralVector()
        
        arrs = [v.to_numpy() for v in vectors]
        weighted_sum = np.zeros(7)
        total_weight = sum(weights)
        
        if total_weight == 0:
            return vectors[0]

        for arr, w in zip(arrs, weights):
            weighted_sum += arr * w
            
        avg_arr = weighted_sum / total_weight
        
        # Merge constraints and reasoning
        all_constraints = []
        reasoning_parts = []
        for v, w in zip(vectors, weights):
            if v.constraints:
                all_constraints.extend(v.constraints)
            if v.reasoning:
                reasoning_parts.append(f"[{w:.2f}] {v.reasoning}")
                
        return MoralVector.from_numpy(
            avg_arr, 
            reasoning=" | ".join(reasoning_parts),
            constraints=list(set(all_constraints))
        )

class MoralTrace(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
