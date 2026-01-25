
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionHead(nn.Module):
    def __init__(self, action_dim, context_dim, hidden_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network - simple MLP for noise prediction
        # Input: action + time_emb + context
        input_dim = action_dim + hidden_dim + context_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, action_dim)
        )

    def forward(self, action, time, context):
        # action: (B, action_dim)
        # time: (B,)
        # context: (B, context_dim)
        
        t_emb = self.time_mlp(time)
        x = torch.cat([action, t_emb, context], dim=1)
        return self.model(x)
