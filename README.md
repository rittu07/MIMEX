# MIMEX

🤖 MIMEX Helrit_bot Portal v2.0
This project is an advanced web-based dashboard designed to control and visualize a 5-DOF (Degree of Freedom) robotic arm system. It acts as the "brain" connecting a physical input device (Leader) to a physical robot (Follower) and provides powerful simulation tools.

Core Capabilities:

📡 Leader-Follower Teleoperation:
Connects to a Leader ESP32 (Input Rig) to read joint angles.
Connects to a Follower ESP32 (Servo Arm) to send movement commands.
Uses PID Algorithms and latency compensation in JavaScript to ensure smooth motion.
🎛️ Manual Web Control:
Allows users to control the robot directly from the browser using sliders.
Calculates Forward Kinematics (FK) to display the exact X, Y, Z coordinates of the gripper.
🧠 AI & Automation:
Neural Policy: A framework to load AI models that can take over control (e.g., for autonomous sorting).
Recording: You can record manual movements and save them as datasets (.csv/.json) for training AI.
🧊 Visualization:
3D Live Preview: A real-time 3D simulation of the robot in an industrial environment (using Three.js).
Kinetic Visualizer: A 2D schematic view for quick status checks.
