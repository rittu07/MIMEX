
import torch
import argparse
import numpy as np
import cv2
import os
import sys

# Ensure we can import backend modules
sys.path.append(os.getcwd())

from backend.mini_vla.models.vla_diffusion_policy import VLADiffusionPolicy
from backend.mini_vla.models.tokenizer import SimpleTokenizer

try:
    import metaworld
except ImportError:
    metaworld = None

def test(args):
    device = torch.device(args.device)
    tokenizer = SimpleTokenizer()
    
    # Determine dimensions
    obs_dim = 39 # Default for push-v3
    action_dim = 4
    
    mt1 = None
    if metaworld is not None:
        from metaworld import MT1
        try:
            mt1 = MT1(args.env_name)
            env = mt1.train_classes[args.env_name]()
            # Setup dimensions from env if possible, else stick to defaults
        except Exception as e:
            print(f"Error loading env: {e}")
            env = None
    else:
        print("Metaworld not installed. Running in dry-run/mock mode.")
        env = None

    print(f"Loading model from {args.checkpoint}...")
    # Ideally should save config with model, but for now using defaults/args
    policy = VLADiffusionPolicy(obs_dim, action_dim, tokenizer.vocab_size).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint)
        policy.eval()
        print("Model loaded.")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Cannot proceed.")
        return

    if env is None:
        print("No environment to test on. Exiting.")
        return

    env.set_task(mt1.train_tasks[0])
    
    frames = []
    
    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        done = False
        steps = 0
        
        print(f"Episode {ep+1}...")
        while not done and steps < args.max_steps:
             # Get Image 
            try:
                img = env.render(mode='rgb_array', width=128, height=128)
            except:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
                
            if img is not None:
                frames.append(img)
                img_in = cv2.resize(img, (64, 64))
            else:
                img_in = np.zeros((64, 64, 3), dtype=np.uint8)
            
            img_in = np.transpose(img_in, (2, 0, 1)) / 255.0
            
            img_tensor = torch.from_numpy(img_in).float().unsqueeze(0).to(device)
            # Use constant instruction
            tokens = torch.tensor(tokenizer.encode(args.instruction), dtype=torch.long).unsqueeze(0).to(device)
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            
            action = policy.predict_action(img_tensor, tokens, state_tensor)
            action = action.cpu().numpy()[0]
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done, truncated, info = step_result
                if truncated: done = True
            
            steps += 1
            
    if args.save_video and len(frames) > 0:
        os.makedirs(args.video_dir, exist_ok=True)
        video_path = os.path.join(args.video_dir, f"{args.env_name}_test.mp4")
        # Save video using cv2
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for f in frames:
            video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) # Convert RGB to BGR for cv2
        video.release()
        
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--env-name', type=str, default='push-v3')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--instruction', type=str, default="push the object to the goal")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos')
    args = parser.parse_args()
    test(args)
