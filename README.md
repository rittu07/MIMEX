# MIMEX Helrit_bot Portal v2.0

## 🤖 Project Overview
**MIMEX Helrit_bot** is an advanced, web-based control interface designed for the teleoperation and simulation of robotic manipulator arms. It serves as the central nervous system for a "Leader-Follower" robotic setup, allowing for real-time synchronization, manual web control, and AI-driven automation.

The dashboard features a high-performance **Cyberpunk / Neon-Industrial UI**, built with responsiveness and visual feedback in mind.

## 🚀 Key Drivers & Features

### 1. Dual-Mode Control System
*   **📡 Follow Leader (Teleoperation):** Real-time synchronization where a "Follower" robotic arm mimics the movements of a "Leader" input arm (sensor array) via WebSocket. Includes PID control and latency compensation for smooth motion.
*   **🎛️ Web Control (Manual):** Direct manipulation of the robot joints (J1-J5) using on-screen sliders or keyboard shortcuts.

### 2. Hardware Integration
*   **Leader ESP32:** Transmits sensor data (potentiometer angles) from the input rig.
*   **Follower ESP32:** Receives servo commands to actuate the physical robot arm.
*   **WebSocket Comms:** Low-latency communication protocol handling 20Hz+ data streams between the browser and microcontrollers.

### 3. Advanced Visualization
*   **🧊 3D Live Preview:** A fully interactive Three.js environment rendering the robot's state in a realistic industrial scene.
*   **📐 Kinetic Visualizer:** A 2D SVG-based schematic providing clear, schematic feedback of arm linkages and gripper status.
*   **📊 Telemetry:** Real-time graphs and readouts for joint angles, TCP (Tool Center Point) coordinates (Forward Kinematics), and velocity metrics.

### 4. Intelligence & Automation
*   **🧠 Neural Policy Control:** A framework for loading AI models (Behavior Cloning) to autonomously control the robot based on learned demonstrations.
*   **🔴 Record & Playback:** robust system to record motion sequences, save them as CSV/JSON datasets, and replay them with adjustable speeds and looping.

## 🛠️ Tech Stack
*   **Frontend:** HTML5, Vanilla JavaScript (ES6+), CSS3 (Custom Neon variables).
*   **Libraries:** `Three.js` (3D Rendering), `Chart.js` (Telemetry).
*   **Connectivity:** Native WebSockets.
*   **Hardware (Firmware):** Arduino/C++ for ESP32/ESP8266.

## 📂 Project Structure
*   `index.html` - The main dashboard structure and layout.
*   `script.js` - The core logic engine (Physics, PIDs, networking, UI events).
*   `style.css` - Visual styling, animations, and responsive grid layouts.
*   `follower_esp32_code.ino` - Firmware for the receiving robot arm.

## 🎯 Use Cases
*   Remote Manipulation / Telepresence.
*   Collecting datasets for Robot Learning (Imitation Learning).
*   Educational demonstration of Forward Kinematics and Control Theory.
