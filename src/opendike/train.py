import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from src.opendike import MoralVector, MoralExpert, MemPalace, LayeredMoralityDeducer
from src.opendike.config import config

logger = logging.getLogger(__name__)

class ContinuousTrainer:
    """
    Orchestrates the training flow for experts and gating networks.
    Supports initializing expert weights from traces and logically segregating traces.
    """
    def __init__(self, deducer: LayeredMoralityDeducer, mem_palace: MemPalace):
        self.deducer = deducer
        self.mem_palace = mem_palace
        self.gating_net = GatingNetwork(
            input_dim=384, 
            num_experts=len(deducer.experts) if deducer.experts else 5
        )

    def segregate_and_assign_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan MemPalace and organize traces by their expert key (layer_type:layer_id).
        Returns a mapping of expert_key -> list of trace data.
        """
        segregated = {}
        for layer_type, layers in self.mem_palace.storage.items():
            for layer_id, traces in layers.items():
                expert_key = f"{layer_type}:{layer_id}"
                segregated[expert_key] = [
                    {
                        "content": t.content,
                        "embedding": t.embedding,
                        "metadata": t.metadata,
                        "timestamp": t.timestamp
                    } for t in traces
                ]
        return segregated

    def initialize_expert_weights(self, expert_key: str, method: str = "centroid"):
        """
        Initialize or re-calibrate an expert's base_profile based on its assigned traces.
        'centroid' method: Adjust base profile towards the mean of its traces' implied deltas.
        """
        expert = self.deducer.experts.get(expert_key)
        if not expert:
            logger.warning(f"Expert {expert_key} not found for initialization.")
            return

        traces = self.mem_palace.storage.get(expert.layer_type, {}).get(expert.layer_id, [])
        if not traces:
            return

        # Extract deltas from traces
        deltas = []
        for t in traces:
            if "vector_delta" in t.metadata:
                deltas.append(np.array(t.metadata["vector_delta"]))
        
        if not deltas:
            return

        # Compute mean delta
        mean_delta = np.mean(deltas, axis=0)
        
        # Update expert base profile
        new_vec = np.clip(expert.base_profile.to_numpy() + mean_delta, 0.0, 1.0)
        expert.base_profile = MoralVector.from_numpy(
            new_vec, 
            reasoning=f"Initialized/Recalibrated from {len(traces)} traces at {datetime.now().isoformat()}"
        )
        logger.info(f"Updated base profile for {expert_key} using {len(traces)} traces.")

    def run_continuous_cycle(self):
        """
        One full cycle: Segregate traces -> Initialize/Update Experts -> (Future) Train Gating.
        """
        logger.info("Starting continuous training cycle...")
        
        # 1. Segregate
        segregated = self.segregate_and_assign_traces()
        
        # 2. Update Experts
        for expert_key in segregated.keys():
            self.initialize_expert_weights(expert_key)
            
        # 3. Gating Training (Simulated)
        # In a real scenario, we'd collect (query, optimal_expert_weights) pairs
        # for batch training here.
        logger.info("Continuous training cycle complete.")

class GatingNetwork(nn.Module):
    """
    Learns to route queries to specific moral experts.
    Input: Query embedding
    Output: Softmax weights over N experts
    """
    def __init__(self, input_dim=384, num_experts=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, query_embedding):
        return self.net(query_embedding)

    def run_training_step(self, query_embedding: torch.Tensor, target_weights: torch.Tensor):
        """
        Update gating network weights given a query and desired routing.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Predicted weights
        pred = self.forward(query_embedding)
        
        # target_weights should be a distribution (softmax already applied)
        loss = criterion(torch.log(pred), target_weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

class SmallMoralTransformer(nn.Module):
    """
    A 125M parameter-style small transformer for nuanced moral deduction.
    Can be used as the backbone for a single expert or the whole deducer.
    """
    def __init__(self, input_dim=384, hidden_dim=768, output_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding):
        # Handle batch or single vector
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        x = self.encoder(embedding)
        x = self.transformer_layers(x)
        x = self.decoder(x)
        return self.sigmoid(x)

def train_moe_step(query_embedding, target_moral_vector, gating_net, experts):
    """
    A single training step for the MoE architecture.
    """
    optimizer_gate = torch.optim.Adam(gating_net.parameters(), lr=1e-3)
    
    # 1. Get routing weights
    routing_weights = gating_net(query_embedding)
    
    # 2. Get expert outputs (simulated here as we use heuristics in core.py for now)
    # In a full training setup, each expert would be a trainable SmallMoralTransformer
    expert_outputs = torch.stack([torch.tensor(e.get_vector("").to_numpy(), dtype=torch.float32) for e in experts])
    
    # 3. Compute MoE output
    moe_output = torch.matmul(routing_weights, expert_outputs)
    
    # 4. Compute Loss (MSE between MoE output and target moral vector)
    loss = nn.MSELoss()(moe_output, target_moral_vector)
    
    # 5. Backprop
    optimizer_gate.zero_grad()
    loss.backward()
    optimizer_gate.step()
    
    return loss.item()

def train_simulation():
    print("=== OpenDike Continuous Training Simulation ===")
    
    # 1. Setup components
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    trainer = ContinuousTrainer(deducer, palace)
    
    # 2. Add some traces to MemPalace to simulate history
    user_id = "user_123"
    print(f"Injecting traces for {user_id}...")
    
    # Trace 1: User wants more authority
    palace.store_trace("personal", user_id, {
        "content": "I want more respect for rules.",
        "metadata": {"vector_delta": [0, 0, 0, 0.2, 0, 0, 0]} # authority boost
    })
    
    # Trace 2: User wants more care
    palace.store_trace("personal", user_id, {
        "content": "Prioritize helping people more.",
        "metadata": {"vector_delta": [0.3, 0, 0, 0, 0, 0, 0]} # care boost
    })
    
    # Register the expert if it doesn't exist (Deducer usually does this on the fly or from config)
    if f"personal:{user_id}" not in deducer.experts:
        deducer.register_expert("personal", user_id, MoralVector.default())
    
    print(f"Initial Personal Expert Profile: {deducer.experts[f'personal:{user_id}'].base_profile.to_dict()['weights']}")
    
    # 3. Run Training Cycle
    print("\nRunning Continuous Training Cycle...")
    trainer.run_continuous_cycle()
    
    # 4. Verify Update
    updated_profile = deducer.experts[f"personal:{user_id}"].base_profile
    print(f"Updated Personal Expert Profile: {updated_profile.to_dict()['weights']}")
    print(f"Reasoning: {updated_profile.reasoning}")

    # 5. Gating Training Step
    print("\nSimulating Gating Network Training Step...")
    # Get current num_experts to avoid mismatch
    num_experts = trainer.gating_net.net[2].out_features
    query_emb = torch.randn(384)
    target_routing = torch.ones(num_experts) / num_experts # Uniform target for simplicity in demo
    loss = trainer.gating_net.run_training_step(query_emb, target_routing)
    print(f"Gating Training Loss: {loss:.4f}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    train_simulation()
