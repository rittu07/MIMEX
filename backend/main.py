
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import threading
import time

# Import VLA Modules
from policy.mlp_policy import load_policy
from vision.encoder import VisionEncoder
from task.task_encoder import encode_task
from kinematics.differentiable_arm import DifferentiableArm

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- GLOBAL STATE ---
CURRENT_MODE = "RULE" # 'RULE' | 'VLA'
latest_joints = [90, 90, 90, 90, 90, 0] # 6 DOF (including gripper)
latest_frame = None

# --- LOAD BRAIN ---
brain = load_policy()
vision_enc = VisionEncoder()
ik_solver = DifferentiableArm()

# --- ROUTES ---

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global CURRENT_MODE
    data = request.json
    CURRENT_MODE = data.get('mode', 'RULE')
    print(f"🔄 Mode Switched to: {CURRENT_MODE}")
    return jsonify({"status": "ok", "mode": CURRENT_MODE})

@app.route('/update_state', methods=['POST'])
def update_state():
    global latest_joints
    # Receive current state from Frontend (Leader Arm / Sim)
    data = request.json
    latest_joints = data.get('joints', latest_joints)
    
    # If in VLA mode, return the POLICY'S calculation
    if CURRENT_MODE == "VLA":
        action = run_vla_inference()
        return jsonify({"mode": "VLA", "target_joints": action})
    
    return jsonify({"mode": "RULE", "message": "Standard manual control active"})

@app.route('/solve_ik', methods=['POST'])
def solve_ik():
    data = request.json
    target = data.get('target', [0, 0.5, 0.5]) # XYZ
    current = data.get('current_joints', [90, 90, 90, 90, 90])
    
    # Run Numerical Solver
    solved_joints, error = ik_solver.inverse_kinematics(target, current)
    
    return jsonify({
        "status": "success", 
        "joints": solved_joints,
        "error": error
    })

# --- INFERENCE LOOP ---
def run_vla_inference():
    # 1. Vision
    vis_feat = vision_enc.get_features(None) # Pass latest_frame in reality
    
    # 2. Task
    task_feat = encode_task("PICK_DROP") # Hardcoded goal for now
    
    # 3. Proprioception
    # Normalize joints 0-180 -> -1 to 1
    j_tensor = torch.tensor([[(j - 90)/90.0 for j in latest_joints]])
    
    # 4. Forward Pass
    with torch.no_grad():
        action_delta = brain(vis_feat, task_feat, j_tensor)
    
    # 5. Denormalize
    # Action is delta (e.g. -0.1 to 0.1 rads/step) or target
    # Let's assume it predicts absolute target (normalized)
    target_norm = action_delta.numpy()[0]
    target_joints = [(t * 90.0) + 90.0 for t in target_norm]
    
    return [int(t) for t in target_joints]

if __name__ == '__main__':
    print("🚀 VLA Backend Server Starting on Port 5000...")
    app.run(host='0.0.0.0', port=5000)
