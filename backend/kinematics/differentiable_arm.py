
import torch
import torch.nn as nn
import numpy as np

class DifferentiableArm(nn.Module):
    def __init__(self, link_lengths=[0.4, 0.4, 0.35, 0.1]):
        super().__init__()
        self.L = link_lengths
    
    def forward(self, theta):
        """
        Forward Kinematics (Differentiable)
        theta: [N, 5] tensor of joint angles in DEGREES
        Returns: [N, 3] tensor of XYZ coordinates
        """
        # Convert to Radians
        rad = torch.deg2rad(theta)
        t1, t2, t3, t4, t5 = rad[:,0], rad[:,1], rad[:,2], rad[:,3], rad[:,4]
        
        # Simplified 4-DOF Planar Model (as used in JS version) + Base Rotation
        # L1 (Base Height) is Z offset
        # L2 (Shoulder), L3 (Elbow), L4 (Wrist)
        # Note: Depending on your DH params, these signs might flip.
        # Matching the logic from script.js:
        # y_coord = L1 + L2 * cos(t2) + L3 * cos(t2 + t3) + L4 * cos(t2 + t3 + t4)
        # r = L2 * sin(t2) + L3 * sin(t2 + t3) + L4 * sin(t2 + t3 + t4)
        # x = r * sin(t1)
        # z = r * cos(t1)
        
        # NOTE: script.js uses (theta - 90) offset!
        # We must align with the physical calibration.
        
        # Offsets (assuming input is raw 0-180 servo range)
        t1 = t1 - np.pi/2
        t2 = t2 - np.pi/2
        t3 = t3 - np.pi/2
        t4 = t4 - np.pi/2
        
        y = self.L[0] + self.L[1]*torch.cos(t2) + self.L[2]*torch.cos(t2+t3) + self.L[3]*torch.cos(t2+t3+t4)
        r = self.L[1]*torch.sin(t2) + self.L[2]*torch.sin(t2+t3) + self.L[3]*torch.sin(t2+t3+t4)
        
        x = r * torch.sin(t1)
        z = r * torch.cos(t1)
        
        return torch.stack([x, y, z], dim=1)

    def inverse_kinematics(self, target_pos, current_joints, steps=50, lr=0.1):
        """
        Numerical IK using Jacobian Descent (via Autograd)
        target_pos: [x, y, z]
        current_joints: [j1, j2, j3, j4, j5]
        """
        target = torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)
        
        # Learnable Joints (Initialize with current state)
        theta = torch.tensor(current_joints, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        
        optimizer = torch.optim.Adam([theta], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Forward Pass
            current_pos = self.forward(theta)
            
            # Loss (MSE Distance)
            loss = torch.sum((current_pos - target)**2)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Joint Limits Constraint (0 - 180)
            with torch.no_grad():
                theta.clamp_(0, 180)
                
            if loss.item() < 0.001:
                break
                
        return theta.detach().numpy()[0].tolist(), loss.item()
