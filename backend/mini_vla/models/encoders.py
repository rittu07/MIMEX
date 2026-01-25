
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # 3 -> 32 -> 64 -> 128
        # Using stride 2 to reduce dimensions by half at each step
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.projection = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global Average Pooling
        # x: (B, 128, H', W') -> (B, 128)
        x = torch.mean(x, dim=[2, 3])
        
        # Projection
        x = self.projection(x)
        return x

class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)
        # s: (B, L, H)
        _, h_n = self.gru(x)
        # h_n: (1, B, H)
        x = h_n.squeeze(0)
        x = self.norm(x)
        return x

class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)
