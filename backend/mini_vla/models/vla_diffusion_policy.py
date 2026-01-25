
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionModule
from .diffusion_head import DiffusionHead

class VLADiffusionPolicy(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 vocab_size, 
                 embed_dim=256, 
                 context_dim=256, 
                 hidden_dim=256,
                 num_timesteps=100):
        super().__init__()
        
        self.image_encoder = ImageEncoderTinyCNN(embed_dim)
        self.text_encoder = TextEncoderTinyGRU(vocab_size, embed_dim)
        self.state_encoder = StateEncoderMLP(state_dim, embed_dim)
        
        self.fusion = FusionModule(embed_dim, context_dim)
        self.diffusion_head = DiffusionHead(action_dim, context_dim, hidden_dim)
        
        self.num_timesteps = num_timesteps
        self.action_dim = action_dim
        
        # Beta schedule
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1.0 - beta
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))
        self.register_buffer('beta', beta)

    def encode(self, image, text_tokens, state):
        v = self.image_encoder(image)
        t = self.text_encoder(text_tokens)
        s = self.state_encoder(state)
        return self.fusion(v, t, s)

    def compute_loss(self, image, text_tokens, state, action):
        # Encode inputs
        context = self.encode(image, text_tokens, state)
        
        B = action.shape[0]
        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (B,), device=action.device).long()
        
        # Sample noise
        noise = torch.randn_like(action)
        
        # Add noise to action (Forward process)
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        alpha_bar_t = self.alpha_bar[t].view(B, 1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        noise_pred = self.diffusion_head(noisy_action, t, context)
        
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, image, text_tokens, state):
        device = image.device
        B = image.shape[0]
        
        context = self.encode(image, text_tokens, state)
        
        # Start from pure noise
        action = torch.randn(B, self.action_dim, device=device)
        
        # Reverse denoising process
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.diffusion_head(action, t, context)
            
            # DDPM Update
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps) + sigma_t * z
            
            alpha_t = self.alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.beta[i]
            
            if i > 0:
                z = torch.randn_like(action)
            else:
                z = 0
                
            action = (1 / torch.sqrt(alpha_t)) * (
                action - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
            ) + torch.sqrt(beta_t) * z
            
        return action
