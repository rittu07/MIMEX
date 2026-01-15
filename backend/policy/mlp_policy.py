
import torch
import torch.nn as nn

class VLABrain(nn.Module):
    def __init__(self, vision_dim=512, task_dim=2, joint_dim=6, action_dim=6):
        super().__init__()
        input_dim = vision_dim + task_dim + joint_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim), # Output: Predicted Delta or Absolute Joint Angles
            nn.Tanh() # Assuming normalized output -1 to 1
        )

    def forward(self, vision, task, joints):
        # Concatenate inputs: [Batch, Vision] + [Batch, Task] + [Batch, Joints]
        x = torch.cat([vision, task, joints], dim=1)
        return self.net(x)

def load_policy(path="vla_checkpoint.pth"):
    model = VLABrain()
    try:
        model.load_state_dict(torch.load(path))
        print(f"✅ Loaded VLA Policy from {path}")
    except:
        print("⚠️ No checkpoint found, initializing random brain.")
    return model
