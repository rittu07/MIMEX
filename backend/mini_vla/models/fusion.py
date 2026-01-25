
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, embed_dim, context_dim=256):
        super().__init__()
        # Input is concat of 3 * embed_dim (vision + text + state)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, context_dim)
        )
    
    def forward(self, v, t, s):
        # v, t, s are (B, embed_dim)
        x = torch.cat([v, t, s], dim=1)
        return self.net(x)
