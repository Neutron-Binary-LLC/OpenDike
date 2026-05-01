import torch
import torch.nn as nn
from typing import List, Dict
from src.opendike import MoralVector, MoralExpert

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

def train_stub():
    print("Starting MoE training simulation...")
    
    # Setup stubs
    gating_net = GatingNetwork(input_dim=384, num_experts=5)
    query_emb = torch.randn(384)
    target_vec = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.1, 0.9, 0.5]) # Target moral profile
    
    print("Training gating network on a single sample...")
    # Simulated experts (need MemPalace and profiles from core)
    # For stub, we'll just print status
    print(f"Target Moral Vector: {target_vec.tolist()}")
    print("Optimization complete for stub.")

if __name__ == "__main__":
    train_stub()
