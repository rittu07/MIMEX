
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import sys

# Ensure we can import backend modules
# Assuming running from root c:\fff
sys.path.append(os.getcwd())

from backend.mini_vla.models.vla_diffusion_policy import VLADiffusionPolicy
from backend.mini_vla.models.tokenizer import SimpleTokenizer

class VLADataset(Dataset):
    def __init__(self, data_path, tokenizer):
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        # Assuming data structure: image, instruction, state, action
        self.images = data['image'] 
        self.instructions = data['instruction']
        self.states = data['state']
        self.actions = data['action']
        self.tokenizer = tokenizer
        print(f"Loaded {len(self.images)} samples.")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Image: (H, W, 3) -> (3, H, W)
        img = self.images[idx]
        # Basic normalization and permute
        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0
        
        # Instruction
        instr = str(self.instructions[idx])
        tokens = torch.tensor(self.tokenizer.encode(instr), dtype=torch.long)
        
        # State
        state = torch.from_numpy(self.states[idx]).float()
        
        # Action
        action = torch.from_numpy(self.actions[idx]).float()
        
        return img, tokens, state, action

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = SimpleTokenizer()
    
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}. Please run collect_data.py first.")
        return

    dataset = VLADataset(args.dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Get dimensions from first batch
    img, tok, state, act = dataset[0]
    state_dim = state.shape[0]
    action_dim = act.shape[0]
    vocab_size = tokenizer.vocab_size
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Vocab Size: {vocab_size}")
    
    policy = VLADiffusionPolicy(state_dim, action_dim, vocab_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    print(f"Starting training...")
    
    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            img_b, tok_b, state_b, act_b = [x.to(device) for x in batch]
            
            loss = policy.compute_loss(img_b, tok_b, state_b, act_b)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(policy.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-path', type=str, default='checkpoints/model.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)
