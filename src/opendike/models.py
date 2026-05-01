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

class MoralTrace(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
