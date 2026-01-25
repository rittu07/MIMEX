
import argparse
import numpy as np
import os
import cv2
import time
import sys

try:
    import metaworld
    import metaworld.policies as policies
except ImportError:
    metaworld = None

def collect_data(args):
    if metaworld is None:
        print("MetaWorld not installed. Cannot collect real data.")
        if args.mock:
            print("Generating MOCK data...")
            # Generating random data for testing the training pipeline
            N = args.episodes * args.max_steps
            data = {
                'image': np.random.randint(0, 255, (N, 64, 64, 3), dtype=np.uint8),
                'instruction': ["push the object to the goal"] * N,
                'state': np.random.randn(N, 39), # Assume 39 dim
                'action': np.random.randn(N, 4)
            }
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            np.savez(args.output_path, 
                     image=data['image'], 
                     instruction=data['instruction'],
                     state=data['state'],
                     action=data['action'])
            print(f"Saved MOCK data to {args.output_path}")
        else:
            print("Run with --mock to generate dummy data for testing.")
        return

    # Check for MT1
    from metaworld import MT1
    mt1 = MT1(args.env_name)
    env = mt1.train_classes[args.env_name]()
    task = mt1.train_tasks[0]
    env.set_task(task)
    
    # Get Policy
    # Assuming policy names follow pattern 'SawyerPushV3Policy' etc.
    policy_name = f"Sawyer{args.env_name.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_cls = getattr(policies, policy_name)
        policy = policy_cls()
    except AttributeError:
        print(f"Could not find policy {policy_name} for {args.env_name}.")
        return

    data = {
        'image': [],
        'instruction': [],
        'state': [],
        'action': []
    }
    
    instruction = "push the object to the goal" # Static for this task
    
    print(f"Collecting {args.episodes} episodes using {policy_name}...")
    
    for ep in range(args.episodes):
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
             obs = reset_res[0]
        else:
             obs = reset_res
            
        done = False
        steps = 0
        
        while not done and steps < args.max_steps:
            # Get Image logic (requires mujoco rendering)
            # Metaworld/Mujoco envs allow render
            try:
                img = env.render(mode='rgb_array', width=128, height=128) 
            except Exception as e:
                print(f"Render failed: {e}")
                img = np.zeros((128, 128, 3), dtype=np.uint8)

            if img is None:
                 img = np.zeros((128, 128, 3), dtype=np.uint8)

            # Get Action
            action = policy.get_action(obs)
            
            # Store
            data['image'].append(cv2.resize(img, (64, 64))) 
            data['instruction'].append(instruction)
            data['state'].append(obs)
            data['action'].append(action)
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done, truncated, info = step_result
                if truncated: done = True
            
            steps += 1
            
        print(f"Episode {ep+1} collected.")
        
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, 
             image=np.array(data['image']), 
             instruction=np.array(data['instruction']),
             state=np.array(data['state']),
             action=np.array(data['action']))
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='push-v3')
    parser.add_argument('--camera-name', type=str, default='corner')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--output-path', type=str, default='data/metaworld_push_bc.npz')
    parser.add_argument('--mock', action='store_true', help="Generate mock data if metaworld is missing")
    args = parser.parse_args()
    collect_data(args)
