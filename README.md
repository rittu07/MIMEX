# MIMEX Helrit_bot Portal v2.5

## 🤖 Project Overview
**MIMEX Helrit_bot** is an advanced, web-based control interface designed for the teleoperation and simulation of robotic manipulator arms. It acts as the central nervous system for a "Leader-Follower" robotic setup and features a **Real-Time Vision-Language-Action (VLA) Backend** for AI-driven autonomy.

The dashboard features a high-performance **Cyberpunk / Neon-Industrial UI**, built with responsiveness and visual feedback in mind.

## 🚀 Key Drivers & Features

### 1. Dual-Mode Control System
*   **📡 Follow Leader (Teleoperation):** Real-time synchronization where a "Follower" robotic arm mimics the movements of a "Leader" input arm (sensor array) via WebSocket. Includes PID control and latency compensation.
*   **🎛️ Web Control (Manual):** Direct manipulation of the robot joints (J1-J5) using on-screen sliders or keyboard shortcuts.

### 2. 🧠 VLA Intelligence (New!)
A dedicated Python backend enabling **Neural Policy Control**:
*   **Vision-Language-Action (VLA) Architecture:** Integrates visual features, task tokens, and joint state into a learned MLP policy.
*   **Backend Inference:** A Flask + PyTorch server processes state updates and returns intelligent servo commands.
*   **Mode Switch:** Seamlessly toggle between "Rule-Based" (Manual) and "VLA-Based" (AI) execution from the UI.

### 3. Hardware Integration
*   **Leader ESP32:** Transmits sensor data (potentiometer angles) from the input rig.
*   **Follower ESP32:** Receives servo commands to actuate the physical robot arm.
*   **WebSocket Comms:** Low-latency communication handling 20Hz+ data streams.

### 4. Advanced Visualization
*   **🧊 3D Live Preview:** A fully interactive Three.js environment with GLTF model support.
*   **📐 Kinetic Visualizer:** A 2D SVG-based schematic providing clear schematic feedback.
*   **📊 Telemetry:** Real-time graphs for joint angles, TCP coordinates, and velocity metrics.

## 🛠️ Tech Stack
*   **Frontend:** HTML5, Vanilla JavaScript (ES6+), CSS3 (Neon UI), Three.js.
*   **Backend (AI):** Python, Flask, PyTorch (VLA Policy).
*   **Connectivity:** Native WebSockets (Hardware), HTTP REST (AI Backend).
*   **Firmware:** Arduino/C++ for ESP32.

## 📂 Project Structure
*   `index.html` - The main dashboard structure.
*   `script.js` - Core logic (Physics, PIDs, UI) and Frontend-Backend bridge.
*   `backend/` - **The VLA Brain**
    *   `main.py` - Flask Server & Inference Loop.
    *   `policy/mlp_policy.py` - PyTorch VLA Model.
    *   `vision/` - Vision Encoders (MobileNet/CLIP placeholders).
*   `assets/` - 3D Models (.glb) for the visualizer.
*   `follower_esp32_code.ino` - Firmware for the receiving robot arm.

## 🎯 Use Cases
*   Remote Manipulation / Telepresence.
*   **Imitation Learning:** Collecting datasets to train the VLA policy.
*   Autonomous Pick-and-Place (using the VLA backend).
