# MIMEX Helrit_bot Portal v2.5

<div align="center">
  <h3>Advanced Robotic Teleoperation & VLA Control Interface</h3>
  <p>A "Leader-Follower" robotic arm system with a Cyberpunk/Sci-Fi Web Dashboard.</p>
</div>

---

## 🤖 Project Overview
**MIMEX Helrit_bot** is a comprehensive robotic control system designed for remote manipulation, data collection, and AI-driven autonomy. It links a physical "Leader" arm (input device) to a "Follower" arm (actuator) via low-latency WebSockets, while providing a rich 3D visualization and simulation environment in the browser.

The system features a **Vision-Language-Action (VLA)** backend integration, allowing the robot to transition from manual teleoperation to learned neural policy execution.
Now updated with a **Professional Laptop Software UI** and **Secure Login System** for enterprise-grade operation.

---

## 🚀 Key Features

### 1. Dual-Mode Control
*   **📡 Follow Leader (Teleoperation):** Real-time synchronization where the Follower arm mimics the Leader inputs.
    *   **Low Latency:** Optimized WebSocket binary protocol (~20Hz).
    *   **PID Stabilization:** Smooths jittery human inputs.
*   **🎛️ Manual Web Control:** Direct control via on-screen sliders or keyboard shortcuts.

### 2. 🧠 VLA Intelligence & Autonomy
*   **Neural Policy Integration:** Toggle between "Classical" and "Learned Policy" modes.
*   **Data Collection:** Record demonstration trajectories to train Behavior Cloning (BC) models.
*   **Export Formats:** Supports CSV, JSON, and LeRobot dataset standards for AI training.

### 3. 🎥 Advanced Visualization
*   **3D Live Preview:** Interactive Three.js scene with realistic physics, lighting/shadows, and environment toggles (Factory, Table, Obstacles).
*   **Kinetic Visualizer:** SVG-based schematic providing clear joint-angle feedback.
*   **Real-Time Telemetry:** Graphs for joint angles, TCP (Tool Center Point) coordinates, and velocity.

### 4. 🎬 Sequence Recorder & Playback
*   **Waypoint Editor:** Create complex motion sequences.
*   **Playback Types:**
    *   **Single Run:** Execute once.
    *   **Loop/Cycle:** Run N times.
    *   **Wait Nodes:** Insert pauses between moves.
*   **Inverse Kinematics (IK):** Supports `MoveL` (Linear) interpolation alongside `MoveJ` (Joint).

---

## 🛠️ Hardware & Pinout

### Leader Arm (Controller)
**Board:** ESP32 Dev Module
*   **J1 (Base):** Potentiometer -> GPIO 32
*   **J2 (Shoulder):** Rotary Encoder -> CLK: GPIO 25, DT: GPIO 26
*   **J3 (Elbow):** Potentiometer -> GPIO 34
*   **J4 (Wrist):** Potentiometer -> GPIO 35
*   **J5 (Gripper):** Potentiometer -> GPIO 39

### Follower Arm (Robot)
**Board:** NodeMCU (ESP8266)
*   **J1 (Base):** Servo -> D4
*   **J2 (Shoulder):** Servo -> D5
*   **J3 (Elbow):** Servo -> D6
*   **J4 (Wrist):** Servo -> D7
*   **J5 (Gripper):** Servo -> D1

*(Note: Connect Servos to separate 5V Power Supply. Do not power directly from ESP8266)*

---

## 📦 Installation & Setup

### 1. Firmware Flash
1.  **Leader:** Open `leader_esp32_code/leader_esp32_code.ino` in Arduino IDE.
    *   Update `ssid` and `password`.
    *   Upload to ESP32.
    *   Note the IP Address printed in Serial Monitor (e.g., `192.168.1.100`).
2.  **Follower:** Open `follower_esp8266_code/follower_esp8266_code.ino`.
    *   Update `ssid` and `password`.
    *   Update `leader_ip` to match the Leader's IP.
    *   Upload to NodeMCU ESP8266.

### 2. Desktop App Setup (Downloadable Software)
To run this as a standalone native application (Windows/Linux/Mac):

1.  **Install Dependencies:**
    ```bash
    npm install
    ```
2.  **Run Application:**
    ```bash
    npm start
    ```
3.  **Build Executable (optional):**
    ```bash
    npm run dist
    ```
    *This will generate a `.exe` or `.dmg` file in the `dist` folder.*

---



---

## 🎮 Controls & Shortcuts
Ensure **"Web Control"** mode is active for keyboard inputs.

| Key | Action | Description |
| :--- | :--- | :--- |
| **A / D** | J1 (Base) | Rotate Left / Right |
| **W / X** | J2 (Shoulder) | Lift Up / Down |
| **E / Z** | J3 (Elbow) | Flex Up / Down |
| **Q / C** | J4 (Wrist) | Pitch Up / Down |
| **P / K** | J5 (Gripper) | Open / Close |
| **S** | STOP | Emergency Stop / Stop Recording |
| **?** | Help | Toggle Help Modal |

---

## 📂 Project Structure
```text
/
├── assets/                 # 3D Models (.glb)
├── backend/                # Python VLA Backend (Flask + PyTorch)
├── follower_esp8266_code/  # ESP8266 Firmware
├── leader_esp32_code/      # ESP32 Firmware
├── index.html              # Main Dashboard
├── script.js               # Core Logic (UI, PID, WebSocket, Three.js)
├── style.css               # Neon/Cyberpunk Styling
└── README.md               # Documentation
```

## 🔧 Troubleshooting
*   **"Unable to handle compilation..."**: Ensure `.ino` files are in their own folders (Fixed in v2.5).
*   **WebSocket Disconnected**: Check if ESP32 and Laptop are on the same WiFi network. Use `ping <ESP_IP>` to verify.
*   **Jerky Motion**: Adjust PID values in `script.js` (`kp` variable) or reduce `updateInterval`.
