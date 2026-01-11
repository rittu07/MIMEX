document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const updateInterval = 10; // ms (100Hz for maximum smoothness)

    // --- DOM Elements ---
    const recordBtn = document.getElementById('record-btn');
    const playbackBtn = document.getElementById('playback-btn');
    const fileInput = document.getElementById('file-input');
    const modeToggle = document.getElementById('mode-toggle');
    const controlBtns = document.querySelectorAll('.ctrl-btn');

    // Arrays for elements
    const bars = [1, 2, 3, 4, 5].map(i => document.getElementById(`bar-${i}`));
    const vals = [1, 2, 3, 4, 5].map(i => document.getElementById(`val-${i}`));
    const sliders = [1, 2, 3, 4, 5].map(i => document.getElementById(`slider-${i}`));
    const disps = [1, 2, 3, 4, 5].map(i => document.getElementById(`disp-${i}`));

    // Visualizer Joints
    const vizJoints = [1, 2, 3, 4].map(i => document.getElementById(`viz-j${i}`)); // J1-J4 rotate entire groups
    const clawLeft = document.getElementById('claw-left');
    const clawRight = document.getElementById('claw-right');

    // Modal Elements (Now Dropdown)
    const recordDropdown = document.getElementById('record-dropdown');
    const waypointNameInput = document.getElementById('waypoint-name');
    const addPointBtn = document.getElementById('add-point-btn');
    const finishBtn = document.getElementById('finish-btn');
    const pointCounter = document.getElementById('point-counter');

    // Playback Dropdown Elements
    const playbackDropdown = document.getElementById('playback-dropdown');
    const playbackLogsPanel = document.getElementById('playback-logs-panel');
    const playbackControlsPanel = document.getElementById('playback-controls-panel');
    const playLastChoiceBtn = document.getElementById('play-last-choice-btn');
    const openFileChoiceBtn = document.getElementById('open-file-choice-btn');
    const undoPointBtn = document.getElementById('undo-point-btn');
    const stepBtn = document.getElementById('btn-step');
    const explainBtn = document.getElementById('btn-explain');
    const programLog = document.getElementById('program-log');
    const systemLog = document.getElementById('system-log');
    const recordedPointsLog = document.getElementById('recorded-points-log');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const threeContainer = document.getElementById('three-container');
    const tcpValX = document.getElementById('tcp-x');
    const tcpValY = document.getElementById('tcp-y');
    const tcpValZ = document.getElementById('tcp-z');
    const fsModeVal = document.getElementById('fs-mode-val');
    const fsStatusVal = document.getElementById('fs-status-val');

    // WebSocket Elements
    const wsUrlInput = document.getElementById('ws-url');
    const wsConnectBtn = document.getElementById('ws-connect-btn');
    const wsStatusDisplay = document.getElementById('ws-status');

    // Follower WebSocket Elements
    const followerWsUrlInput = document.getElementById('follower-ws-url');
    const followerConnectBtn = document.getElementById('follower-connect-btn');
    const followerStatusDisplay = document.getElementById('follower-status');

    // Execution Mode Elements
    const modeClassicalBtn = document.getElementById('mode-classical');
    const modePolicyBtn = document.getElementById('mode-policy');
    const policyActiveMsg = document.getElementById('policy-active-msg');
    const neuralPolicyPanel = document.getElementById('neural-policy-panel');
    const policyModelInput = document.getElementById('policy-model-input');
    const policyModelNameDisplay = document.getElementById('policy-model-name');
    const policyStatusInd = document.getElementById('policy-status-ind');
    const policyInputVal = document.getElementById('policy-input-val');
    const policyOutputVal = document.getElementById('policy-output-val');
    const policyConf = document.getElementById('policy-conf');
    const startPolicyBtn = document.getElementById('start-policy-btn');
    const trainPolicyBtn = document.getElementById('train-policy-btn');
    const trainStatus = document.getElementById('train-status');

    const VLA_API = "http://localhost:5000"; // Python Backend URL

    // --- State ---
    let currentMode = 'LEADER'; // 'LEADER' | 'WEB' | 'PLAYBACK'
    let executionType = 'CLASSICAL'; // 'CLASSICAL' | 'POLICY'
    let activePolicyModel = null; // Stores loaded model object (mock for now)
    let isPolicyRunning = false; // New flag for START/STOP

    // WebSocket State
    let ws = null;
    let wsConnected = false;
    let wsLatestData = null; // Expect [J1, J2, J3, J4, J5]
    let lastWsTime = 0;
    let jointVelocities = [0, 0, 0, 0, 0]; // For Predictive/Latency Comp

    // Follower WebSocket State
    let followerWs = null;
    let followerConnected = false;
    let lastFollowerSendTime = 0;

    let currentValues = [90, 90, 90, 90, 90]; // Start at 90 deg (0-180)

    // Velocity State
    let previousValues = [90, 90, 90, 90, 90];
    let lastUpdateTime = Date.now();
    const velocityThreshold = 100; // deg/s (Arbitrary threshold for "Jerky")

    let previousMode = 'LEADER';
    let playbackInterval = null;
    let playbackSpeed = 1.0;

    // Data Storage
    let stepIndex = 0;
    let sessionWaypoints = [];
    let allWaypoints = [];

    // --- PID Controller ---
    class PIDController {
        constructor(kp, ki, kd) {
            this.kp = kp;
            this.ki = ki;
            this.kd = kd;
            this.prevError = 0;
            this.integral = 0;
            this.lastTime = Date.now();
        }

        update(target, current) {
            const now = Date.now();
            const dt = (now - this.lastTime) / 1000; // seconds
            this.lastTime = now;

            if (dt <= 0 || dt > 0.5) return current;

            let error = target - current;

            // Deadband Filter: Ignore microscopic noise for stability
            if (Math.abs(error) < 0.3) {
                error = 0;
                this.integral = 0; // Prevent integral windup on noise
                return current; // Lock position
            }

            this.integral += error * dt;
            const derivative = (error - this.prevError) / dt;
            this.prevError = error;

            const output = (this.kp * error) + (this.ki * this.integral) + (this.kd * derivative);
            return current + (output * dt);
        }

        reset() {
            this.integral = 0;
            this.prevError = 0;
            this.lastTime = Date.now();
        }
    }

    // High-Performance Tuning: Faster Catch-up (Kp=12), Little Damping (Kd=0.2)
    const pids = [
        new PIDController(12.0, 0.5, 0.2), // J1
        new PIDController(12.0, 0.5, 0.2), // J2
        new PIDController(12.0, 0.5, 0.2), // J3
        new PIDController(12.0, 0.5, 0.2), // J4
        new PIDController(12.0, 0.5, 0.2)  // J5
    ];

    // --- Simulation Engine (MIMEX) ---
    function generateSensorData() {
        return currentValues.map(prev => {
            const base = 90;
            const jitter = Math.floor(Math.random() * 10) - 5; // Smaller jitter for degrees
            return Math.max(0, Math.min(180, base + jitter));
        });
    }

    const errs = [1, 2, 3, 4, 5].map(i => document.getElementById(`err-${i}`));

    // Velocity UI Elements
    const maxVelDisplay = document.getElementById('max-vel');
    const motionWarning = document.getElementById('motion-warning');

    function calculateVelocity(newValues) {
        const now = Date.now();
        const dt = (now - lastUpdateTime) / 1000; // seconds
        if (dt <= 0) return 0;

        let maxV = 0;
        newValues.forEach((val, i) => {
            const dTheta = Math.abs(val - previousValues[i]);
            const velocity = dTheta / dt; // deg/s
            if (velocity > maxV) maxV = velocity;
        });

        // Update UI
        if (maxVelDisplay) maxVelDisplay.textContent = Math.round(maxV);

        if (motionWarning) {
            if (maxV > velocityThreshold) {
                motionWarning.classList.remove('hidden');
            } else {
                motionWarning.classList.add('hidden');
            }
        }

        // Update History
        previousValues = [...newValues];
        lastUpdateTime = now;

        return maxV;
    }

    function updateDisplay(values) {
        // Calculate Velocity First
        calculateVelocity(values);

        values.forEach((val, index) => {
            if (bars[index]) {
                const percent = (val / 180) * 100; // Scale 0-180 -> 0-100%
                bars[index].style.width = `${percent}%`;
            }
            if (vals[index]) {
                vals[index].textContent = Math.round(val); // Integer degrees
            }
            if (disps[index]) {
                disps[index].textContent = Math.round(val) + '°';
            }
            if (sliders[index]) {
                sliders[index].value = Math.round(val);
            }

            // Deviation Logic (Only in Leader Mode)
            if (errs[index]) {
                if (currentMode === 'LEADER') {
                    // Simulate a "Target" vs "actual" lag for demo purposes
                    const drift = (Math.random() * 2 - 1).toFixed(1);
                    const isPos = drift > 0;
                    errs[index].textContent = (isPos ? '+' : '') + drift + '°';
                    errs[index].className = 'deviation ' + (isPos ? 'positive' : 'negative');
                } else {
                    errs[index].textContent = '--';
                    errs[index].className = 'deviation';
                }
            }
        });

        // --- Update Kinetic Visualizer ---

        // J1: Base Rotation (Yaw). Visualized as pseudo-3D foreshortening.
        // 0 = Left profile, 90 = Frontal (foreshortened), 180 = Right profile.
        // We use scaleX to simulate turning towards/away from camera.
        const vizBaseRot = document.getElementById('viz-base-rot');
        if (vizBaseRot) {
            // Map 0-180 to scaleX 1 -> 0 -> 1 ?
            // Let's say 0 deg = Full Side View (Scale 1)
            // 90 deg = Facing Camera (Scale 0.1)
            // 180 deg = Full Side View Other side (Scale -1 or 1)
            // Let's model it roughly: cos(angle)
            const rad = (values[0] - 90) * (Math.PI / 180);
            const scaleFactor = 0.6 + 0.4 * Math.sin((values[0] / 180) * Math.PI);
            vizBaseRot.setAttribute('transform', `translate(0, -50) scale(${scaleFactor}, 1)`);
        }

        // J2: Shoulder (Old J1).
        if (vizJoints[0]) vizJoints[0].setAttribute('transform', `rotate(${values[1] - 90})`);

        // J3: Elbow (Old J2).
        if (vizJoints[1]) vizJoints[1].setAttribute('transform', `translate(0, -120) rotate(${values[2] - 90})`);

        // J4: Wrist Pitch (Old J3).
        if (vizJoints[2]) vizJoints[2].setAttribute('transform', `translate(0, -100) rotate(${values[3] - 90})`);

        // Wrist Roll (Old J4) - REMOVED ROTATION
        // It just holds the gripper now. We keep it fixed.
        if (vizJoints[3]) vizJoints[3].setAttribute('transform', `translate(0, -60) rotate(0)`);

        // J5: Gripper. Open/Close.
        if (clawLeft && clawRight) {
            const openAngle = (values[4] / 180) * 45; // Max 45 deg open
            clawLeft.setAttribute('transform', `rotate(${-openAngle})`);
            clawRight.setAttribute('transform', `rotate(${openAngle})`);
        }

        // --- Calculate Forward Kinematics (XYZ) ---
        calculateFK(values);
    }

    // Real VLA Inference (Replaces Mock)
    function runVLAPolicy() {
        if (!isPolicyRunning) return;

        // Prepare Payload
        const payload = {
            joints: currentValues
        };

        // Async Fetch (Fire and Forget style to not block UI, but handle response)
        fetch(`${VLA_API}/update_state`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.mode === 'VLA' && data.target_joints) {
                    // Smoothly interpolate to target (Simple P-control or direct set)
                    // For now, let's treat these as direct targets for the PIDs to handle
                    // or just set them if we trust the network speed.

                    // Let's use the PID controller to move towards the AI's target
                    const aiTargets = data.target_joints;

                    // Update UI text
                    if (policyInputVal) policyInputVal.textContent = "[" + currentValues.map(v => Math.round(v)).join(',') + "]";
                    if (policyOutputVal) policyOutputVal.textContent = "[" + aiTargets.map(v => Math.round(v)).join(',') + "]";

                    // Apply to simulation
                    currentValues = currentValues.map((curr, i) => {
                        // Use existing PID logic to smooth out the network jitter
                        // Assume AI runs at ~10Hz, Loop runs at 100Hz
                        return pids[i].update(aiTargets[i], curr);
                    });

                    updateDisplay(currentValues);
                }
            })
            .catch(err => {
                // console.warn("VLA Backend Offline?", err);
                // Fallback to mock if offline? No, let's just show error status
                if (policyStatusInd) {
                    policyStatusInd.textContent = "OFFLINE";
                    policyStatusInd.style.color = "var(--neon-red)";
                }
            });
    }

    // Update Loop
    setInterval(() => {
        if (executionType === 'POLICY') {
            if (isPolicyRunning) {
                runVLAPolicy();
            }
        } else if (currentMode === 'LEADER') {
            // Priority: WebSocket Data > Simulation
            if (wsConnected && wsLatestData) {
                // PID Tracking for smooth motion
                const now = Date.now();
                const predDt = Math.min((now - lastWsTime) / 1000, 0.06); // Cap prediction at 60ms latency

                currentValues = currentValues.map((curr, i) => {
                    const rawTarget = wsLatestData[i] || 90;
                    const vel = jointVelocities[i] || 0;

                    // Latency Compensation: Teleport target into the future
                    let predictedTarget = rawTarget + (vel * predDt);

                    // Clamp prediction to physics limits
                    predictedTarget = Math.max(0, Math.min(180, predictedTarget));

                    return pids[i].update(predictedTarget, curr);
                });
            } else {
                // Reset PIDs if signal lost to prevent windup
                pids.forEach(pid => pid.reset());
            }
            // (Removed commented out junk)

            updateDisplay(currentValues);
        } else if (currentMode === 'WEB') {
            // In WEB mode, we just display the current manual values
            // (which are updated via keys)
            updateDisplay(currentValues);
        }
        // Update Trajectory Plot
        if (typeof updateChartData === 'function') {
            updateChartData();
        }

        // --- Follower Sync ---
        // Send commands to Follower ESP32 if connected
        if (followerConnected && followerWs && followerWs.readyState === WebSocket.OPEN) {
            const now = Date.now();
            if (now - lastFollowerSendTime > 50) { // Limit to 20Hz to prevent flooding
                // Send standard JSON format: {servos: [90, 90, 90, 90, 90]}
                const msg = JSON.stringify({ servos: currentValues.map(v => Math.round(v)) });
                try {
                    followerWs.send(msg);
                } catch (e) {
                    console.error("Follower Send Error:", e);
                }
                lastFollowerSendTime = now;
            }
        }
    }, updateInterval);

    // --- Follower Connection Logic ---
    if (followerConnectBtn) {
        followerConnectBtn.addEventListener('click', () => {
            if (followerConnected) {
                // Disconnect
                if (followerWs) {
                    followerWs.close();
                    followerWs = null;
                }
                followerConnected = false;
                followerConnectBtn.textContent = "📡 Link";
                followerConnectBtn.style.color = "var(--neon-green)";
                followerConnectBtn.style.borderColor = "var(--neon-green)";
                followerStatusDisplay.textContent = "OFFLINE";
                followerStatusDisplay.style.color = "var(--neon-red)";
                logSystem("Follower Disconnected.");
            } else {
                // Connect
                const url = followerWsUrlInput.value;
                logSystem(`Connecting to Follower at ${url}...`);
                followerStatusDisplay.textContent = "CONNECTING...";
                followerStatusDisplay.style.color = "yellow";

                try {
                    followerWs = new WebSocket(url);

                    followerWs.onopen = () => {
                        followerConnected = true;
                        followerConnectBtn.textContent = "❌ Unlink";
                        followerConnectBtn.style.color = "var(--neon-red)";
                        followerConnectBtn.style.borderColor = "var(--neon-red)";
                        followerStatusDisplay.textContent = "LINKED";
                        followerStatusDisplay.style.color = "var(--neon-green)";
                        logSystem("Follower Connected Successfully!");
                    };

                    followerWs.onclose = () => {
                        followerConnected = false;
                        followerConnectBtn.textContent = "📡 Link";
                        followerConnectBtn.style.color = "var(--neon-green)";
                        followerConnectBtn.style.borderColor = "var(--neon-green)";
                        followerStatusDisplay.textContent = "OFFLINE";
                        followerStatusDisplay.style.color = "var(--neon-red)";
                        if (followerWs) logSystem("Follower Connection Closed.");
                        followerWs = null;
                    };

                    followerWs.onerror = (e) => {
                        logSystem("Follower WebSocket Error.");
                        console.error(e);
                        followerStatusDisplay.textContent = "ERROR";
                        followerStatusDisplay.style.color = "var(--neon-red)";
                    };

                    // Optional: Handle incoming messages from Follower if needed
                    // followerWs.onmessage = (msg) => { ... }

                } catch (e) {
                    logSystem("Invalid WebSocket URL for Follower.");
                    console.error(e);
                    followerStatusDisplay.textContent = "ERROR";
                }
            }
        });
    }

    // --- Mode Indicator Update Logic ---
    const modeIndicator = document.getElementById('mode-indicator');

    function updateModeUI() {
        if (!modeIndicator) return;

        // Reset Styles
        modeIndicator.removeAttribute('data-status');
        modeIndicator.style.borderColor = "";
        modeIndicator.style.color = "";

        // Priority 0: Policy Mode (Overrides all)
        if (typeof executionType !== 'undefined' && executionType === 'POLICY') {
            modeIndicator.textContent = "MODE: LEARNED POLICY (AI)";
            modeIndicator.style.borderColor = "var(--neon-green)";
            modeIndicator.style.color = "var(--neon-green)";
            return;
        }

        // Priority 1: Playback
        if (currentMode === 'PLAYBACK') {
            modeIndicator.textContent = "MODE: PLAYBACK";
            modeIndicator.setAttribute('data-status', 'playback');
            return;
        }

        // Priority 2: Recording (Can happen in WEB or LEADER)
        const isRecording = recordBtn.classList.contains('recording');
        if (isRecording) {
            modeIndicator.textContent = "MODE: RECORDING";
            modeIndicator.setAttribute('data-status', 'recording');
            return;
        }

        // Priority 3: Base Modes
        if (currentMode === 'WEB') {
            modeIndicator.textContent = "MODE: MANUAL (Web Control)";
        } else {
            modeIndicator.textContent = "MODE: FOLLOW (Live Teleoperation)";
        }
    }

    function setExecutionMode(type) {
        executionType = type;

        // Notify Backend
        fetch(`${VLA_API}/set_mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: type === 'POLICY' ? 'VLA' : 'RULE' })
        }).catch(e => console.log("Backend offline"));

        if (type === 'CLASSICAL') {
            // UI Updates
            if (modeClassicalBtn) {
                modeClassicalBtn.style.borderColor = "var(--neon-blue)";
                modeClassicalBtn.style.color = "var(--neon-blue)";
                modeClassicalBtn.style.fontWeight = "bold";
            }
            if (modePolicyBtn) {
                modePolicyBtn.style.borderColor = "rgba(255,255,255,0.2)";
                modePolicyBtn.style.color = "rgba(255,255,255,0.5)";
                modePolicyBtn.style.fontWeight = "normal";
            }
            if (policyActiveMsg) policyActiveMsg.style.display = "none";
            if (neuralPolicyPanel) neuralPolicyPanel.classList.add('hidden');

            // Re-enable playback controls
            if (playbackBtn) {
                playbackBtn.style.opacity = "1";
                playbackBtn.style.pointerEvents = "auto";
                playbackBtn.textContent = "PLAYBACK OPTIONS";
            }

        } else {
            // POLICY
            if (modeClassicalBtn) {
                modeClassicalBtn.style.borderColor = "rgba(255,255,255,0.2)";
                modeClassicalBtn.style.color = "rgba(255,255,255,0.5)";
                modeClassicalBtn.style.fontWeight = "normal";
            }
            if (modePolicyBtn) {
                modePolicyBtn.style.borderColor = "var(--neon-green)";
                modePolicyBtn.style.color = "var(--neon-green)";
                modePolicyBtn.style.fontWeight = "bold";
            }
            if (policyActiveMsg) policyActiveMsg.style.display = "block";

            if (activePolicyModel) {
                if (isPolicyRunning) {
                    if (policyStatusInd) {
                        policyStatusInd.textContent = "RUNNING";
                        policyStatusInd.style.color = "var(--neon-green)";
                    }
                } else {
                    if (policyStatusInd) {
                        policyStatusInd.textContent = "READY";
                        policyStatusInd.style.color = "white";
                    }
                }
            } else {
                if (policyStatusInd) {
                    policyStatusInd.textContent = "NO MODEL";
                    policyStatusInd.style.color = "var(--neon-red)";
                }
            }

            if (neuralPolicyPanel) neuralPolicyPanel.classList.remove('hidden');

            // Disable playback controls
            if (playbackBtn) {
                playbackBtn.style.opacity = "0.5";
                playbackBtn.style.pointerEvents = "none";
                playbackBtn.textContent = "PLAYBACK DISABLED";
            }

            // Ensure playback dropdowns are closed
            if (playbackLogsPanel) playbackLogsPanel.classList.add('hidden');
            if (playbackControlsPanel) playbackControlsPanel.classList.add('hidden');
        }

        updateModeUI();

        // Also update full screen overlay if active
        if (fsModeVal) {
            let displayMode = currentMode;
            if (executionType === 'POLICY') displayMode = 'LEARNED POLICY (AI)';
            else if (currentMode === 'LEADER') displayMode = 'FOLLOW LEADER (TELEOP)';
            else if (currentMode === 'WEB') displayMode = 'MANUAL (WEB)';
            fsModeVal.textContent = displayMode;
        }
    }

    if (modeClassicalBtn && modePolicyBtn) {
        modeClassicalBtn.addEventListener('click', () => setExecutionMode('CLASSICAL'));
        modePolicyBtn.addEventListener('click', () => setExecutionMode('POLICY'));
    }

    // --- Model Loading Logic ---
    if (policyModelInput) {
        policyModelInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function (e) {
                try {
                    // Placeholder for actual model parsing (ONNX/JSON)
                    // For now, we just store the name and pretend it's loaded
                    activePolicyModel = {
                        name: file.name,
                        data: e.target.result // Keep raw data for later
                    };

                    logSystem(`Model loaded: ${file.name}`);

                    if (policyModelNameDisplay) {
                        policyModelNameDisplay.textContent = file.name;
                        policyModelNameDisplay.style.color = "var(--neon-green)";
                    }

                    if (executionType === 'POLICY') {
                        if (policyStatusInd) {
                            if (isPolicyRunning) {
                                policyStatusInd.textContent = "RUNNING";
                                policyStatusInd.style.color = "var(--neon-green)";
                            } else {
                                policyStatusInd.textContent = "READY";
                                policyStatusInd.style.color = "white";
                            }
                        }
                    }

                } catch (err) {
                    alert("Failed to load model: " + err.message);
                }
            };
            reader.readAsText(file);
        });
    }

    // --- Training Logic (Mock) ---
    if (trainPolicyBtn) {
        trainPolicyBtn.addEventListener('click', () => {
            if (sessionWaypoints.length === 0) {
                alert("No demonstrations recorded! Please record some points in Classical Mode first.");
                return;
            }

            trainStatus.textContent = "Training... (2s)";
            trainStatus.style.color = "yellow";
            trainPolicyBtn.disabled = true;

            setTimeout(() => {
                trainStatus.textContent = "Model Trained (v1)";
                trainStatus.style.color = "var(--neon-green)";
                trainPolicyBtn.disabled = false;

                // Auto-load the "trained" model
                activePolicyModel = {
                    name: "BC_Model_v1 (Trained)",
                    data: "mock_weights"
                };

                if (policyModelNameDisplay) {
                    policyModelNameDisplay.textContent = activePolicyModel.name;
                    policyModelNameDisplay.style.color = "var(--neon-green)";
                }

                logSystem("Training Complete. Model loaded automatically.");
                alert("Behavior Cloning Training Complete!\nThe policy has learned from your " + sessionWaypoints.length + " demonstration points.");

            }, 2000);
        });
    }



    // --- Policy Start/Stop Logic ---
    if (startPolicyBtn) {
        startPolicyBtn.addEventListener('click', () => {
            if (!activePolicyModel) {
                alert("Please load a model first.");
                return;
            }

            isPolicyRunning = !isPolicyRunning;

            if (isPolicyRunning) {
                startPolicyBtn.textContent = "STOP POLICY";
                startPolicyBtn.style.borderColor = "var(--neon-red)";
                startPolicyBtn.style.color = "var(--neon-red)";
                startPolicyBtn.classList.add('pulse-border'); // Add visual flare

                if (policyStatusInd) {
                    policyStatusInd.textContent = "RUNNING";
                    policyStatusInd.style.color = "var(--neon-green)";
                }
                logSystem("Neural Policy Control STARTED.");

                // Disable Waypoint/Action buttons
                if (recordBtn) recordBtn.disabled = true;
                if (addPointBtn) addPointBtn.disabled = true;
                document.querySelectorAll('.btn-action').forEach(b => {
                    if (b.id !== 'start-policy-btn') b.style.opacity = '0.5';
                });

            } else {
                startPolicyBtn.textContent = "START POLICY (AI)";
                startPolicyBtn.style.borderColor = "var(--neon-green)";
                startPolicyBtn.style.color = "var(--neon-green)";
                startPolicyBtn.classList.remove('pulse-border');

                if (policyStatusInd) {
                    policyStatusInd.textContent = "READY";
                    policyStatusInd.style.color = "white";
                }
                logSystem("Neural Policy Control STOPPED.");

                // Re-enable Waypoint/Action buttons
                if (recordBtn) recordBtn.disabled = false;
                if (addPointBtn) addPointBtn.disabled = false;
                document.querySelectorAll('.btn-action').forEach(b => {
                    b.style.opacity = '1';
                });

                // Keep Playback disabled if in Policy Mode
                if (playbackBtn && executionType === 'POLICY') {
                    playbackBtn.style.opacity = "0.5";
                }
            }
        });
    }

    // --- Mode Toggle Logic ---
    modeToggle.addEventListener('change', () => {
        if (currentMode === 'PLAYBACK') return; // Locked during playback

        if (modeToggle.checked) {
            currentMode = 'WEB';
            console.log("Switched to WEB CONTROL Mode");
            document.querySelector('.header-controls').style.borderColor = 'var(--neon-green)';
            currentValues = [90, 90, 90, 90, 90];
        } else {
            currentMode = 'LEADER';
            console.log("Switched to LEADER Mode");
            document.querySelector('.header-controls').style.borderColor = 'var(--neon-blue)';
        }

        // Update Fullscreen Overlay Text
        if (fsModeVal) {
            let displayMode = currentMode;
            if (executionType === 'POLICY') displayMode = 'LEARNED POLICY (AI)';
            else if (currentMode === 'LEADER') displayMode = 'FOLLOW LEADER (TELEOP)';
            else if (currentMode === 'WEB') displayMode = 'MANUAL (WEB)';
            fsModeVal.textContent = displayMode;
        }
        updateModeUI();
    });

    // --- Record / Waypoint Logic ---
    function startRecording() {
        sessionWaypoints = [];
        pointCounter.textContent = "0";
        updateRecordedPointsLog();
        waypointNameInput.value = "";

        // Clear Metadata fields logic if needed, but keeping them might be useful?
        // Let's clear them for a fresh start or keep last? Let's keep for convenience. 
        // document.getElementById('meta-task').value = ""; ...

        recordDropdown.classList.remove('hidden');
        waypointNameInput.focus();
        recordBtn.textContent = "STOP RECORDING";
        recordBtn.classList.add('recording');
        updateModeUI(); // Update Indicator
        logSystem("Recording session started.");
        updateProgramLog(sessionWaypoints);
    }

    function stopRecording() {
        recordDropdown.classList.add('hidden');
        recordBtn.textContent = "RECORD SEQUENCE";
        recordBtn.classList.remove('recording');
        updateModeUI(); // Update Indicator
        logSystem("Recording session stopped. " + sessionWaypoints.length + " points saved.");
        updateProgramLog(sessionWaypoints);

        // Clear Live Path
        if (robotParts.pathGroup) {
            while (robotParts.pathGroup.children.length > 0) {
                robotParts.pathGroup.remove(robotParts.pathGroup.children[0]);
            }
        }
    }

    // --- Home Logic ---
    const homeBtn = document.getElementById('home-btn');
    if (homeBtn) {
        homeBtn.addEventListener('click', () => {
            // Only allow if in Manual (WEB) mode to avoid conflict with Leader
            if (currentMode !== 'WEB') {
                alert("Switch to MANUAL (Web Control) Mode to use Home function.");
                return;
            }

            // Smooth Interpolation to Home
            const targetValues = [90, 90, 90, 90, 90];
            const duration = 1500; // 1.5 seconds for smooth motion
            const startValues = [...currentValues];
            const startTime = Date.now();

            homeBtn.textContent = "MOVING...";
            homeBtn.disabled = true;

            // Clear any existing playback interval to prevent conflict (though mode check handles most)
            if (playbackInterval) clearInterval(playbackInterval);

            const homeInterval = setInterval(() => {
                const elapsed = Date.now() - startTime;
                // Ease-out function for smoother stop: 1 - (1 - t)^2
                let t = Math.min(elapsed / duration, 1);
                let easeT = 1 - Math.pow(1 - t, 2);

                currentValues = startValues.map((start, i) => start + (targetValues[i] - start) * easeT);

                updateDisplay(currentValues);

                if (t >= 1) {
                    clearInterval(homeInterval);
                    currentValues = [...targetValues]; // Snap to exact
                    updateDisplay(currentValues);
                    homeBtn.textContent = "GO TO HOME";
                    homeBtn.disabled = false;
                }
            }, 16); // ~60 FPS
        });
    }

    recordBtn.addEventListener('click', () => {
        const isDropdownOpen = !recordDropdown.classList.contains('hidden');
        if (isDropdownOpen) stopRecording();
        else startRecording();
    });

    // --- Waypoint Motion UI Logic ---
    const wpSpeedSlider = document.getElementById('wp-speed');
    const wpSpeedVal = document.getElementById('wp-speed-val');
    if (wpSpeedSlider) {
        wpSpeedSlider.addEventListener('input', (e) => {
            if (wpSpeedVal) wpSpeedVal.textContent = e.target.value;
        });
    }

    addPointBtn.addEventListener('click', () => {
        const name = waypointNameInput.value || `Point ${sessionWaypoints.length + 1}`;

        // Capture Metadata
        const task = document.getElementById('meta-task').value || "undefined";
        const env = document.getElementById('meta-env').value || "undefined";
        const notes = document.getElementById('meta-notes').value || "";

        // Capture Motion
        const motionElement = document.getElementById('wp-motion');
        const speedElement = document.getElementById('wp-speed');

        const motionType = motionElement ? motionElement.value : 'MoveJ';
        const speed = speedElement ? parseInt(speedElement.value) : 60;

        const pointData = {
            type: 'WAYPOINT',
            name: name,
            timestamp: new Date().toISOString(),
            mode: currentMode,
            metadata: {
                task: task,
                environment: env,
                notes: notes
            },
            motion: {
                type: motionType,
                speed: speed
            },
            sensors: [...currentValues],
            coords: calculateFK(currentValues) // Store cartesian for MoveL
        };
        sessionWaypoints.push(pointData);
        pointCounter.textContent = sessionWaypoints.length;
        waypointNameInput.value = "";
        waypointNameInput.focus();

        // Live Trajectory Visualization
        if (typeof drawTrajectory === 'function') {
            drawTrajectory(sessionWaypoints);
        }

        addPointBtn.textContent = "Added!";
        setTimeout(() => addPointBtn.textContent = "Add Waypoint & Continue", 500);

        logSystem(`Waypoint ${sessionWaypoints.length} added: ${name}`);
        updateProgramLog(sessionWaypoints);
        updateRecordedPointsLog();
    });

    // --- Undo Logic ---
    if (undoPointBtn) {
        undoPointBtn.addEventListener('click', () => {
            if (sessionWaypoints.length > 0) {
                const removed = sessionWaypoints.pop();
                pointCounter.textContent = sessionWaypoints.length;
                logSystem(`Waypoint removed: ${removed.name}`);

                // Redraw Trajectory
                if (typeof drawTrajectory === 'function') {
                    drawTrajectory(sessionWaypoints);
                }
                updateProgramLog(sessionWaypoints);
                updateRecordedPointsLog();
            } else {
                logSystem("Nothing to undo.");
            }
        });
    }

    function updateRecordedPointsLog() {
        if (!recordedPointsLog) return;
        recordedPointsLog.innerHTML = "";

        if (sessionWaypoints.length === 0) {
            recordedPointsLog.innerHTML = '<div style="color: grey; font-style: italic;">No points recorded yet.</div>';
            return;
        }

        sessionWaypoints.forEach(p => {
            const vals = p.sensors.map(v => Math.round(v));
            const div = document.createElement('div');
            // Format [90, 142, 39, 134, 42]
            div.textContent = `[${vals.join(', ')}]`;
            recordedPointsLog.appendChild(div);
        });
        recordedPointsLog.scrollTop = recordedPointsLog.scrollHeight;
    }

    // Update Fullscreen Overlay
    if (fsModeVal) {
        let displayMode = currentMode;
        if (currentMode === 'LEADER') displayMode = 'FOLLOW LEADER (TELEOP)';
        if (currentMode === 'WEB') displayMode = 'MANUAL (WEB)';
        fsModeVal.textContent = displayMode;
    }

    // --- Fullscreen Logic (Browser API) ---
    if (fullscreenBtn && threeContainer) {
        fullscreenBtn.addEventListener('click', () => {
            const card = threeContainer.closest('.card');
            if (card) {
                if (!document.fullscreenElement) {
                    card.requestFullscreen().catch(err => {
                        alert(`Error: ${err.message}`);
                    });
                } else {
                    document.exitFullscreen();
                }
            }
        });

        // Listen for Fullscreen change events
        document.addEventListener('fullscreenchange', () => {
            const card = threeContainer.closest('.card');
            if (document.fullscreenElement) {
                card.classList.add('fullscreen-active');
                // Force resize after short delay
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            } else {
                card.classList.remove('fullscreen-active');
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            }
        });
    }

    // --- View Settings Logic ---
    const viewSettingsBtn = document.getElementById('view-settings-btn');
    const viewSettingsPanel = document.getElementById('view-settings-panel');
    const toggleFloor = document.getElementById('toggle-floor');
    const toggleTable = document.getElementById('toggle-table');
    const toggleObstacles = document.getElementById('toggle-obstacles');
    const toggleShadows = document.getElementById('toggle-shadows');

    if (viewSettingsBtn && viewSettingsPanel) {
        viewSettingsBtn.addEventListener('click', () => {
            if (viewSettingsPanel.classList.contains('hidden')) {
                viewSettingsPanel.classList.remove('hidden');
            } else {
                viewSettingsPanel.classList.add('hidden');
            }
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (!viewSettingsPanel.contains(e.target) && e.target !== viewSettingsBtn) {
                viewSettingsPanel.classList.add('hidden');
            }
        });
    }

    // Toggle Listeners
    if (toggleFloor) toggleFloor.addEventListener('change', updateSceneObjects);
    if (toggleTable) toggleTable.addEventListener('change', updateSceneObjects);
    if (toggleObstacles) toggleObstacles.addEventListener('change', updateSceneObjects);
    if (toggleShadows) toggleShadows.addEventListener('change', updateSceneObjects);

    // --- Log Helpers ---
    function logSystem(msg) {
        if (fsStatusVal) fsStatusVal.textContent = msg; // Update FS Overlay status too
        if (!systemLog) return;
        const time = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.innerHTML = `<span style="color: rgba(255,255,255,0.5);">[${time}]</span> ${msg}`;
        systemLog.appendChild(line);
        systemLog.scrollTop = systemLog.scrollHeight;
    }

    function updateProgramLog(points) {
        if (!programLog) return;
        programLog.innerHTML = "";
        points.forEach((p, i) => {
            const div = document.createElement('div');
            // Format: 1. MoveJ [Base] -> [90,90,90,90,90]
            // Simplified for display
            const vals = p.sensors.map(v => Math.round(v)).join(',');
            div.textContent = `${i + 1}. ${p.motion.type} [Base] -> [${vals}]`;
            programLog.appendChild(div);
        });
        programLog.scrollTop = programLog.scrollHeight;
    }

    // --- Playback Extra Buttons ---
    if (stepBtn) {
        stepBtn.addEventListener('click', () => {
            logSystem("Stepping through program...");
            // Placeholder for step logic
            if (sessionWaypoints.length > 0) playSequence(sessionWaypoints, true); // True = Step Mode
            else alert("No recorded points.");
        });
    }

    if (explainBtn) {
        explainBtn.addEventListener('click', () => {
            logSystem("Requesting Neural Explanation...");
            alert("Neural Policy Explanation:\n\nThe current trajectory is optimized for minimal jerk and energy efficiency based on the learned latent policy z.");
        });
    }

    // --- Inverse Kinematics (Geometric 2-Link) ---
    function solveIK(targetX, targetY, targetZ, currentJoints) {
        const L1 = 0.4;
        const L2 = 0.4;
        const L3 = 0.45; // Forearm + Hand approx

        // J1: Yaw
        // atan2(x, z) gives angle from Z axis. 
        // In our visualizer: x = r*sin(t1), z = r*cos(t1). 
        // So t1 = atan2(x, z). 
        // 90 deg = 0 rad. 
        // So J1 = (t1 * 180/PI) + 90.
        let j1 = Math.atan2(targetX, targetZ) * (180 / Math.PI) + 90;

        // Planar Projection
        const r_target = Math.sqrt(targetX * targetX + targetZ * targetZ);
        const y_target = targetY - L1; // Height relative to shoulder

        // Cosine Law for 2-link (L2, L3)
        const D = Math.sqrt(r_target * r_target + y_target * y_target);

        // Check Reachability
        if (D > (L2 + L3) || D < Math.abs(L2 - L3) || D === 0) {
            return null; // Unreachable
        }

        // Angle of D vector from horizontal
        // atan2(y, r)
        const alpha = Math.atan2(y_target, r_target);

        // Internal angle at Shoulder (Gamma)
        // L3^2 = L2^2 + D^2 - 2*L2*D*cos(gamma)
        // cos(gamma) = (L2^2 + D^2 - L3^2) / (2*L2*D)
        const cosGamma = (L2 * L2 + D * D - L3 * L3) / (2 * L2 * D);
        const gamma = Math.acos(Math.min(1, Math.max(-1, cosGamma)));

        // Shoulder Pitch (J2)
        // Relative to horizontal? 
        // My FK: y = L2*cos(t2)... so t2=0 is UP.
        // My alpha is angle from horizontal. Up is alpha=PI/2.
        // So angle from UP = (PI/2 - alpha).
        // Then we subtract gamma? 
        // Let's assume Elbow Up configuration.
        // Angle of L2 relative to D is -gamma?
        // Angle of L2 from horizontal = alpha + gamma. 
        // Angle from UP = PI/2 - (alpha + gamma).

        // Let's rely on standard convention.
        // t2 is angle of L2 relative to vertical (0=Up).
        // alpha_vertical = atan2(r, y)
        const alpha_v = Math.atan2(r_target, y_target);

        // t2 = alpha_v - gamma (Elbow Up/Back) vs alpha_v + gamma (Elbow Down/Forward)
        // Let's try minus.
        const t2_rad = alpha_v - gamma;

        // Internal angle at Elbow (Beta)
        // D^2 = L2^2 + L3^2 - 2*L2*L3*cos(beta')
        // cos(beta') = (L2^2 + L3^2 - D^2) / (2*L2*L3)
        const cosBeta = (L2 * L2 + L3 * L3 - D * D) / (2 * L2 * L3);
        const beta = Math.acos(Math.min(1, Math.max(-1, cosBeta)));

        // t3 is relative to L2 line. 0 means straight.
        // In our FK: t3 adds to t2. 
        // Standard geometric: t3_angle = PI - beta?
        // Actually simple check: values[2] (J3) 90 is straight? 
        // FK: t3 = (val-90). 90->0.
        // Yes.
        // External angle deflection is PI - Internal(beta).
        const t3_rad = Math.PI - beta;

        // Convert to Degrees
        let j2 = 90 + t2_rad * (180 / Math.PI);
        let j3 = 90 + t3_rad * (180 / Math.PI);

        // Constraints
        j1 = Math.max(0, Math.min(180, j1));
        j2 = Math.max(0, Math.min(180, j2));
        j3 = Math.max(0, Math.min(180, j3));

        return [j1, j2, j3, currentJoints[3], currentJoints[4]];
    }

    // --- Playback Logic (Refactored) ---
    function playSequence(data, stepMode = false) {
        if (!data || data.length === 0) {
            alert("No data to play!");
            return;
        }

        // Cycle Management
        const cycleEl = document.getElementById('playback-cycles');
        let totalCycles = cycleEl ? (parseInt(cycleEl.value) || 1) : 1;
        let currentCycle = 0;

        // Step Mode Initialization
        if (stepMode) {
            if (stepIndex >= data.length) {
                stepIndex = 0; // Wrap to start
            }
        } else {
            stepIndex = 0; // Reset for full playback
        }

        if (playbackInterval) clearInterval(playbackInterval);

        previousMode = currentMode;
        currentMode = 'PLAYBACK';

        if (stepMode) {
            playbackBtn.textContent = `STEP ${stepIndex + 1}/${data.length}`;
        } else {
            playbackBtn.textContent = `CYCLE 1/${totalCycles}`;
        }

        playbackBtn.disabled = true;

        const cancelBtn = document.getElementById('cancel-playback-btn');
        if (cancelBtn) cancelBtn.classList.remove('hidden');

        // Draw Planned Path (Visual context)
        if (typeof drawTrajectory === 'function') {
            drawTrajectory(data);
        }

        // Status UI
        const statusPanel = document.getElementById('playback-status');
        const pbWp = document.getElementById('pb-wp');
        const pbMotion = document.getElementById('pb-motion');
        const pbSpeed = document.getElementById('pb-speed');
        const pbCycle = document.getElementById('pb-cycle');

        if (statusPanel) statusPanel.classList.remove('hidden');

        updateModeUI();

        let waypointIndex = stepMode ? stepIndex : 0;
        let segmentStartTime = Date.now();
        let isMovingToHome = false;

        // Start State
        let startJoints = [...currentValues];
        let startCoords = calculateFK(startJoints);
        const homePos = [90, 90, 90, 90, 0];

        // Setup Target for first segment (for logging)
        if (data.length > 0) {
            let targetPoint = data[waypointIndex];
            const startName = "Current Pos";
            const targetName = targetPoint.name || `WP ${waypointIndex + 1}`;
            if (stepMode) logSystem(`Stepping: ${startName} -> ${targetName}`);
            else logSystem(`Playback: ${startName} -> ${targetName}`);
        }

        const frameRate = 20; // 20ms update

        playbackInterval = setInterval(() => {
            // Check Completion
            if (!isMovingToHome && waypointIndex >= data.length) {
                if (stepMode) {
                    finishPlayback();
                    return;
                }

                currentCycle++;
                if (currentCycle >= totalCycles) {
                    finishPlayback();
                    alert("Playback Complete!");
                    return;
                } else {
                    // Trigger Return to Home
                    isMovingToHome = true;
                    logSystem(`Cycle ${currentCycle} Done. Returning to HOME...`);

                    segmentStartTime = Date.now();
                    startJoints = [...currentValues];
                    startCoords = calculateFK(startJoints);
                    return;
                }
            }

            // Safety Check
            if (!isMovingToHome && waypointIndex >= data.length) return;

            // Define Target Logic
            let finalTargetSensors;
            let motionType = 'MoveJ';
            let speedPct = 50;
            let currentTargetName = "";
            let targetPoint = null;

            if (isMovingToHome) {
                finalTargetSensors = homePos;
                motionType = 'MoveJ';
                speedPct = 60;
                currentTargetName = "HOME";
            } else {
                targetPoint = data[waypointIndex];
                finalTargetSensors = targetPoint.sensors;
                motionType = targetPoint.motion ? targetPoint.motion.type : 'MoveJ';
                speedPct = targetPoint.motion ? targetPoint.motion.speed : 50;
                currentTargetName = targetPoint.name || `WP ${waypointIndex + 1}`;
            }

            // Update Status UI Loop
            if (pbWp) pbWp.textContent = currentTargetName;
            if (pbMotion) pbMotion.textContent = motionType;
            if (pbSpeed) pbSpeed.textContent = speedPct;
            if (pbCycle) pbCycle.textContent = stepMode ? "-" : `${currentCycle + 1}/${totalCycles}`;

            // Duration
            const baseTime = 2000;
            const duration = (baseTime / (speedPct / 100)) / playbackSpeed;

            const now = Date.now();
            const elapsed = now - segmentStartTime;
            let progress = elapsed / duration;

            // Segment Complete?
            if (progress >= 1) {
                // Snap to target
                currentValues = [...finalTargetSensors];
                updateDisplay(currentValues);

                if (isMovingToHome) {
                    isMovingToHome = false;
                    waypointIndex = 0;
                    playbackBtn.textContent = `CYCLE ${currentCycle + 1}/${totalCycles}`;
                    logSystem(`Starting Cycle ${currentCycle + 1}...`);

                    segmentStartTime = Date.now();
                    startJoints = [...currentValues];
                    startCoords = calculateFK(startJoints);

                    if (data.length > 0) {
                        const nextPt = data[0];
                        logSystem(`Moving to ${nextPt.name || 'WP 1'}`);
                    }
                    return;
                }

                // Advance Regular Waypoint
                waypointIndex++;

                if (stepMode) {
                    stepIndex = waypointIndex;
                    finishPlayback();
                    return;
                }

                if (waypointIndex < data.length) {
                    const nextName = data[waypointIndex].name || `WP ${waypointIndex + 1}`;
                    logSystem(`Moving to ${nextName}`);
                }

                segmentStartTime = Date.now();
                startJoints = [...currentValues];
                startCoords = calculateFK(startJoints);
                return;
            }

            // Interpolate
            if (motionType === 'MoveJ') {
                currentValues = startJoints.map((s, i) => {
                    const t = finalTargetSensors[i];
                    return s + (t - s) * progress;
                });
            } else if (motionType === 'MoveL') {
                const targetCoords = (targetPoint && !isMovingToHome) ? (targetPoint.coords || calculateFK(targetPoint.sensors)) : calculateFK(homePos);

                const curX = startCoords.x + (targetCoords.x - startCoords.x) * progress;
                const curY = startCoords.y + (targetCoords.y - startCoords.y) * progress;
                const curZ = startCoords.z + (targetCoords.z - startCoords.z) * progress;

                const solved = solveIK(curX, curY, curZ, startJoints);
                if (solved) {
                    currentValues = solved;
                } else {
                    currentValues = startJoints.map((s, i) => {
                        const t = finalTargetSensors[i];
                        return s + (t - s) * progress;
                    });
                }
            }

            updateDisplay(currentValues);

        }, frameRate);

        function finishPlayback() {
            clearInterval(playbackInterval);
            playbackInterval = null;
            currentMode = previousMode;
            // Restore button text
            playbackBtn.textContent = "PLAY LAST";
            playbackBtn.disabled = false;

            if (cancelBtn) cancelBtn.classList.add('hidden');
            if (statusPanel) statusPanel.classList.add('hidden');

            updateModeUI();
        }
    }

    // --- Export Logic ---
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const posX = document.getElementById('pos-x'); // New elements
    const posY = document.getElementById('pos-y');
    const posZ = document.getElementById('pos-z');

    function checkPoints() {
        if (sessionWaypoints.length === 0) {
            alert("No points added to session!");
            return false;
        }
        return true;
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', () => {
            if (!checkPoints()) return;
            // Append to global history
            allWaypoints = [...allWaypoints, ...sessionWaypoints];
            exportToCSV(sessionWaypoints);
            stopRecording();
        });
    }

    if (exportJsonBtn) {
        exportJsonBtn.addEventListener('click', () => {
            if (!checkPoints()) return;
            allWaypoints = [...allWaypoints, ...sessionWaypoints];
            exportToJSON(sessionWaypoints);
            stopRecording();
        });
    }

    function exportToJSON(data) {
        const jsonContent = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonContent], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const link = document.createElement("a");
        link.href = url;
        link.download = "MIMEX_Demonstration_" + Date.now() + ".json";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // --- Cancel Playback Logic ---
    const cancelPlaybackBtn = document.getElementById('cancel-playback-btn');
    if (cancelPlaybackBtn) {
        cancelPlaybackBtn.addEventListener('click', () => {
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = null;
                currentMode = previousMode;
                playbackBtn.textContent = "PLAY LAST";
                playbackBtn.disabled = false;
                cancelPlaybackBtn.classList.add('hidden'); // Hide self
                alert("Playback Cancelled.");
                updateModeUI();
            }
        });
    }

    // --- Playback Logic ---

    const speedSlider = document.getElementById('speed-slider');
    const speedValDisplay = document.getElementById('speed-val');

    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            playbackSpeed = parseFloat(e.target.value);
            speedValDisplay.textContent = playbackSpeed.toFixed(1) + '×';
        });
    }



    function exportToCSV(data) {
        // Updated Header with Metadata columns
        let csvContent = "data:text/csv;charset=utf-8,Type,Name,Timestamp,Mode,Task,Environment,Notes,J1,J2,J3,J4,J5\n";

        data.forEach(row => {
            const sensors = row.sensors.join(",");
            const meta = row.metadata || { task: '', environment: '', notes: '' };
            // Escape commas in notes
            const safeNotes = `"${meta.notes.replace(/"/g, '""')}"`;
            csvContent += `${row.type},${row.name},${row.timestamp},${row.mode},${meta.task},${meta.environment},${safeNotes},${sensors}\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "MIMEX_Flight_Path_" + Date.now() + ".csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // --- Forward Kinematics (FK) Calculation ---
    // Simplified planer model for "Kinetic Visualizer" logic
    function calculateFK(joints) {
        const L1 = 0.4;
        const L2 = 0.4;
        const L3 = 0.35;
        const L4 = 0.1;

        const t1 = (joints[0] - 90) * (Math.PI / 180);
        const t2 = (joints[1] - 90) * (Math.PI / 180);
        const t3 = (joints[2] - 90) * (Math.PI / 180);
        const t4 = (joints[3] - 90) * (Math.PI / 180);

        const y_coord = L1 + L2 * Math.cos(t2) + L3 * Math.cos(t2 + t3) + L4 * Math.cos(t2 + t3 + t4);
        const r_coord = L2 * Math.sin(t2) + L3 * Math.sin(t2 + t3) + L4 * Math.sin(t2 + t3 + t4);

        const x_coord = r_coord * Math.sin(t1);
        const z_coord = r_coord * Math.cos(t1);

        if (posX) posX.textContent = x_coord.toFixed(2);
        if (posY) posY.textContent = y_coord.toFixed(2);
        if (posZ) posZ.textContent = z_coord.toFixed(2);

        return { x: x_coord, y: y_coord, z: z_coord };
    }

    // --- Controls ---
    function handleWebCommand(cmd) {
        if (cmd === 'center-circle') cmd = 'STOP';

        if (cmd === 'STOP') {
            stopRecording();
            console.log("STOP Command");
            return;
        }

        if (currentMode !== 'WEB') return;

        const step = 5;
        let nextValues = [...currentValues];

        if (cmd === 'J1+') nextValues[0] = Math.min(180, currentValues[0] + step);
        if (cmd === 'J1-') nextValues[0] = Math.max(0, currentValues[0] - step);

        if (cmd === 'J2+') nextValues[1] = Math.min(180, currentValues[1] + step);
        if (cmd === 'J2-') nextValues[1] = Math.max(0, currentValues[1] - step);

        if (cmd === 'J3+') nextValues[2] = Math.min(180, currentValues[2] + step);
        if (cmd === 'J3-') nextValues[2] = Math.max(0, currentValues[2] - step);

        if (cmd === 'J4+') nextValues[3] = Math.min(180, currentValues[3] + step);
        if (cmd === 'J4-') nextValues[3] = Math.max(0, currentValues[3] - step);

        if (cmd === 'J5+') nextValues[4] = Math.min(180, currentValues[4] + step);
        if (cmd === 'J5-') nextValues[4] = Math.max(0, currentValues[4] - step);

        // Collision Check (Via centralized checker)
        if (!checkSafety(nextValues)) {
            logSystem("⚠ Collision Warning: Limit Reached!");
            return; // Block movement
        }

        currentValues = nextValues;
        updateDisplay(currentValues);
    }

    controlBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            handleWebCommand(btn.classList[1]);
        });
    });

    const keyMap = {
        'd': 'J1+', 'a': 'J1-',
        'w': 'J2+', 'x': 'J2-',
        'e': 'J3+', 'z': 'J3-',
        'q': 'J4+', 'c': 'J4-',
        'p': 'J5+', 'k': 'J5-',
        's': 'STOP'
    };

    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        if (key === 'f') {
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = null;
                currentMode = previousMode;
                playbackBtn.disabled = false;
                playbackBtn.textContent = "PLAYBACK";
                if (cancelPlaybackBtn) cancelPlaybackBtn.classList.add('hidden');
                updateModeUI();
            }
            fileInput.click();
        } else if (keyMap[key]) {
            handleWebCommand(keyMap[key]);
        }
    });

    // Event Listeners for Sliders (Manual Update)
    sliders.forEach((slider, index) => {
        if (!slider) return;

        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);

            // Allow manual override in WEB mode OR if in PLAYBACK mode (Paused/Stepping)
            const canOverride = (currentMode === 'WEB' || currentMode === 'PLAYBACK');

            if (canOverride) {
                let testValues = [...currentValues];
                testValues[index] = val;

                // Collision Check (Via centralized checker)
                if (!checkSafety(testValues)) {
                    if (errs[index]) errs[index].textContent = "LIMIT";
                    return;
                }

                currentValues[index] = val;

                // Update specific elements immediately
                if (vals[index]) vals[index].textContent = Math.round(val);
                if (disps[index]) disps[index].textContent = Math.round(val) + '°';
                if (bars[index]) {
                    const percent = (val / 180) * 100;
                    bars[index].style.width = `${percent}%`;
                }

                // Trigger full model update
                updateDisplay(currentValues);
            }
        });
    });

    // --- Safety Checker ---
    function checkSafety(values) {
        // Calculate FK
        const fk = calculateFK(values);

        // Safety Limits
        // 1. Table/Floor Collision
        // Visual observation suggests gripper (L=15cm) + Wrist offset puts tip much lower than FK point.
        // We set a conservative limit to prevent any part from entering the table (y=0 in 3D).
        // If FK returns "Wait J4" height, we need buffer.
        const SAFE_Y_LIMIT = 0.25; // Increased to 25cm

        if (fk.y < SAFE_Y_LIMIT) {
            // Visual Warning
            if (motionWarning) {
                motionWarning.textContent = "⚠ TOO LOW";
                motionWarning.classList.remove('hidden');
                setTimeout(() => motionWarning.classList.add('hidden'), 1000);
            }
            return false;
        }
        return true;
    }

    function moveToTarget(coords) {
        // coords in IK units (meters)
        // Solve IK
        const sol = solveIK(coords.x, coords.y, coords.z, currentValues);
        if (sol) {
            // Animate to it
            logSystem("Moving to Target...");
            // Create a mini-trajectory
            const targetPt = {
                type: 'WAYPOINT',
                name: 'Target',
                sensors: sol,
                motion: { type: 'MoveJ', speed: 40 }
            };
            playSequence([targetPt]);
        } else {
            alert("Target Unreachable!");
        }
    }

    // --- Predefined Moves Handlers (Updated for HUD) ---
    const hudHomeBtn = document.getElementById('hud-home-btn');
    const hudCenterBtn = document.getElementById('hud-center-btn');
    const hudCornerBtn = document.getElementById('hud-corner-btn');
    const hudMidBtn = document.getElementById('hud-mid-btn');

    // Legacy mapping (if element exists)
    const moveCenterBtn = document.getElementById('move-center-btn');
    const moveEdgeBtn = document.getElementById('move-edge-btn');

    if (hudHomeBtn) hudHomeBtn.addEventListener('click', () => { if (homeBtn) homeBtn.click(); });

    // Center: Straight forward, mid reach
    if (hudCenterBtn) hudCenterBtn.addEventListener('click', () => { moveToTarget({ x: 0, y: 0.3, z: 0.5 }); });

    // Corner: Diagonal max reach (approx)
    if (hudCornerBtn) hudCornerBtn.addEventListener('click', () => { moveToTarget({ x: 0.4, y: 0.3, z: 0.4 }); });

    // Mid-Edge: Between Center and Corner
    if (hudMidBtn) hudMidBtn.addEventListener('click', () => { moveToTarget({ x: 0.2, y: 0.3, z: 0.45 }); });

    // Legacy Support
    if (moveCenterBtn) moveCenterBtn.addEventListener('click', () => { moveToTarget({ x: 0, y: 0.3, z: 0.5 }); });
    if (moveEdgeBtn) moveEdgeBtn.addEventListener('click', () => { moveToTarget({ x: 0.4, y: 0.3, z: 0.4 }); });

    const objUp = document.getElementById('obj-up');
    const objDown = document.getElementById('obj-down');
    const objLeft = document.getElementById('obj-left');
    const objRight = document.getElementById('obj-right');
    const objReset = document.getElementById('obj-reset');

    function moveObject(dx, dz) {
        if (!targetBlock) return;
        targetBlock.position.x += dx;
        targetBlock.position.z += dz;
    }

    if (objUp) objUp.addEventListener('click', () => moveObject(-1, 0)); // -X direction (Back)
    if (objDown) objDown.addEventListener('click', () => moveObject(1, 0)); // +X direction (Front)
    if (objLeft) objLeft.addEventListener('click', () => moveObject(0, -1)); // -Z direction (Left)
    if (objRight) objRight.addEventListener('click', () => moveObject(0, 1)); // +Z direction (Right)
    if (objReset) objReset.addEventListener('click', resetTargetBlock);

    // --- Robot Base Controls ---
    const baseUp = document.getElementById('base-up');
    const baseDown = document.getElementById('base-down');
    const baseLeft = document.getElementById('base-left');
    const baseRight = document.getElementById('base-right');
    const baseReset = document.getElementById('base-reset');

    function moveBase(dx, dz) {
        if (!robotParts.root) return;
        robotParts.root.position.x += dx;
        robotParts.root.position.z += dz;
    }

    if (baseUp) baseUp.addEventListener('click', () => moveBase(-1, 0));
    if (baseDown) baseDown.addEventListener('click', () => moveBase(1, 0));
    if (baseLeft) baseLeft.addEventListener('click', () => moveBase(0, -1));
    if (baseRight) baseRight.addEventListener('click', () => moveBase(0, 1));
    if (baseReset) baseReset.addEventListener('click', () => {
        if (robotParts.root) robotParts.root.position.set(0, 0, 0);
    });


    if (baseReset) baseReset.addEventListener('click', () => {
        if (robotParts.root) robotParts.root.position.set(0, 0, 0);
    });

    // --- WebSocket Logic (ESP32 Leader) ---
    if (wsConnectBtn) {
        wsConnectBtn.addEventListener('click', () => {
            if (wsConnected) {
                // Disconnect
                if (ws) ws.close();
            } else {
                // Connect
                const url = wsUrlInput.value.trim();
                connectWebSocket(url);
            }
        });
    }

    function connectWebSocket(inputUrl) {
        let url = inputUrl.trim();
        if (!url) {
            alert("Please enter a valid WebSocket URL");
            return;
        }

        // Auto-fix URL format
        // 1. Add Protocol if missing
        if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
            url = "ws://" + url;
        }


        // Update input box to show formatted URL
        if (wsUrlInput) {
            wsUrlInput.value = url;
        }

        // Check for Mixed Content (HTTPS -> WS)
        if (window.location.protocol === 'https:' && url.startsWith('ws://')) {
            const msg = "Security Warning: Browsers block insecure WebSocket (ws://) from Secure Pages (https://). Please load this page via HTTP or file://, or use WSS.";
            logSystem("❌ " + msg);
            alert(msg);
        }

        wsStatusDisplay.textContent = "CONNECTING...";
        wsStatusDisplay.style.color = "yellow";
        wsConnectBtn.disabled = true;

        try {
            ws = new WebSocket(url);

            ws.onopen = () => {
                wsConnected = true;
                wsStatusDisplay.textContent = "CONNECTED (ONLINE)";
                wsStatusDisplay.style.color = "var(--neon-green)";
                wsConnectBtn.textContent = "❌ Disconnect";
                wsConnectBtn.disabled = false;
                logSystem(`Connected to Leader Arm at ${url}`);
            };

            ws.onmessage = (event) => {
                // Debug: Log raw data
                // console.log("RX:", event.data); 

                let parsed = null;
                try {
                    // Try JSON [v1, v2...]
                    parsed = JSON.parse(event.data);
                } catch (e) {
                    // Fallback: Try Comma-Separated "v1,v2,v3,v4"
                    if (typeof event.data === 'string' && event.data.includes(',')) {
                        parsed = event.data.split(',').map(Number);
                    }
                }

                if (Array.isArray(parsed)) {
                    // Normalize to 5 axes
                    let values = parsed.map(v => {
                        let n = parseFloat(v);
                        return isNaN(n) ? 90 : Math.max(0, Math.min(180, n));
                    });

                    // Pad if 4 axes
                    if (values.length === 4) values.push(0); // Gripper default

                    if (values.length >= 5) {
                        // Velocity Calculation for Prediction
                        const now = Date.now();
                        if (wsLatestData && lastWsTime > 0) {
                            const dt = (now - lastWsTime) / 1000;
                            if (dt > 0.005) { // Avoid div by zero
                                jointVelocities = values.map((v, i) => {
                                    const diff = v - wsLatestData[i];
                                    // Low-pass filter velocity slightly to reduce spike noise
                                    const rawVel = diff / dt;
                                    const oldVel = jointVelocities[i] || 0;
                                    return (oldVel * 0.5) + (rawVel * 0.5);
                                });
                            }
                        }
                        lastWsTime = now;
                        wsLatestData = values;
                    }
                }
            };

            ws.onclose = (event) => {
                wsConnected = false;
                wsLatestData = null;
                wsStatusDisplay.textContent = `DISCONNECTED (${event.code})`;
                wsStatusDisplay.style.color = "var(--neon-red)";
                wsConnectBtn.textContent = "🔌 Connect";
                wsConnectBtn.disabled = false;

                let reason = "Connection Closed";
                if (event.code === 1006) reason = "Connection Refused/Failed (Is IP correct?)";
                logSystem(`${reason} (Code: ${event.code})`);
            };

            ws.onerror = (err) => {
                // WebSocket 'error' event gives no details for security reasons
                console.error("WS Error:", err);
                wsConnected = false;
                wsStatusDisplay.textContent = "ERROR";
                wsStatusDisplay.style.color = "var(--neon-red)";
                wsConnectBtn.disabled = false;
                logSystem("❌ WebSocket Error. Check Console.");
            };

        } catch (e) {
            alert("Error creating WebSocket: " + e.message);
            wsConnectBtn.disabled = false;
        }
    }

    // --- Playback Dropdown Interaction ---
    playbackBtn.addEventListener('click', () => {
        const isHidden = playbackDropdown.classList.contains('hidden');
        if (isHidden) {
            playbackDropdown.classList.remove('hidden');
        } else {
            playbackDropdown.classList.add('hidden');
        }
    });

    playLastChoiceBtn.addEventListener('click', () => {
        playbackDropdown.classList.add('hidden');
        const dataToPlay = sessionWaypoints.length > 0 ? sessionWaypoints : allWaypoints;
        playSequence(dataToPlay);
    });

    openFileChoiceBtn.addEventListener('click', () => {
        playbackDropdown.classList.add('hidden');
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            const parsedData = parseCSV(text);
            if (parsedData.length > 0) {
                playSequence(parsedData);
            } else {
                alert("Could not parse CSV!");
            }
        };
        reader.readAsText(file);
        fileInput.value = '';
    });

    function parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const result = [];
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(',');
            if (cols.length >= 9) {
                // Determine format. If 9+ columns, might be new format with metadata
                // Check if Type is present. 
                // Old: Type,Name,TS,Mode,J1...
                // New: Type,Name,TS,Mode,Task,Env,Notes,J1...
                // Safest is to count from end? J1 is at -5?
                // Let's assume standard index.
                // If col 4 is 'Task' string? No, col 3 is Mode.

                // Let's handle new format:
                // Type[0], Name[1], TS[2], Mode[3], Task[4], Env[5], Notes[6], J1[7], J2[8], J3[9], J4[10], J5[11]

                // Old format:
                // Type[0]... Mode[3], J1[4]...

                let sensors;
                if (isNaN(cols[4])) {
                    // Column 4 is NOT a number (e.g. "pick_place"), so it's metadata
                    sensors = cols.slice(7, 12).map(Number);
                } else {
                    // Column 4 is a number (J1), old format
                    sensors = cols.slice(4, 9).map(Number);
                }

                // Note: Not storing metadata on import for now, just sensors
                result.push({
                    type: cols[0],
                    name: cols[1],
                    mode: cols[3],
                    sensors: sensors
                });
            }
        }
        return result;
    }

    // --- Trajectory Timeline Logic (Chart.js) ---
    const ctx = document.getElementById('trajectory-chart').getContext('2d');
    let trajectoryChart;
    let selectedJointIndex = -1; // -1 = All, 0-4 = Specific
    const maxDataPoints = 50;
    let chartDataBuffer = [];

    // Color Palette for J1-J5
    const jointColors = [
        '#00f3ff', // J1: Blue
        '#00ff41', // J2: Green
        '#ff003c', // J3: Red
        '#f3ff00', // J4: Yellow
        '#bd00ff'  // J5: Purple
    ];
    const jointNames = ["J1 (Base)", "J2 (Shoulder)", "J3 (Elbow)", "J4 (Wrist)", "J5 (Gripper)"];

    function initChart() {
        if (trajectoryChart) trajectoryChart.destroy();

        Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
        Chart.defaults.borderColor = 'rgba(0, 243, 255, 0.2)';

        // Create datasets for ALL joints initially
        const datasets = jointNames.map((name, index) => ({
            label: name,
            data: [],
            borderColor: jointColors[index],
            backgroundColor: jointColors[index] + '1A', // 10% opacity hex
            borderWidth: 2,
            tension: 0.4,
            fill: false,
            pointRadius: 0,
            hidden: false // Visible by default
        }));

        trajectoryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { display: false }
                    },
                    y: {
                        min: 0,
                        max: 180,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            boxWidth: 10,
                            font: { size: 10 }
                        }
                    },
                    tooltip: { displayColors: true }
                }
            }
        });

        // Joint Selector Listener
        const selector = document.getElementById('joint-select');
        // Remove old listener to prevent duplicates (simple clone replacement)
        const newSelector = selector.cloneNode(true);
        selector.parentNode.replaceChild(newSelector, selector);

        newSelector.addEventListener('change', (e) => {
            selectedJointIndex = parseInt(e.target.value);
            refreshChartVisibility();
        });
    }

    function updateChartData() {
        if (!trajectoryChart) return;

        const now = new Date();
        const timeLabel = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`;

        // Push new data frame to buffer
        const newFrame = {
            time: timeLabel,
            values: [...currentValues]
        };
        chartDataBuffer.push(newFrame);

        if (chartDataBuffer.length > maxDataPoints) {
            chartDataBuffer.shift();
        }

        // Update Chart Datasets
        // We push data to ALL datasets regardless of visibility, so history is preserved
        trajectoryChart.data.labels.push(timeLabel);

        for (let i = 0; i < 5; i++) {
            if (trajectoryChart.data.datasets[i]) {
                trajectoryChart.data.datasets[i].data.push(currentValues[i]);
            }
        }

        // Shift chart to scroll
        if (trajectoryChart.data.labels.length > maxDataPoints) {
            trajectoryChart.data.labels.shift();
            for (let i = 0; i < 5; i++) {
                if (trajectoryChart.data.datasets[i]) {
                    trajectoryChart.data.datasets[i].data.shift();
                }
            }
        }

        trajectoryChart.update();
    }

    function refreshChartVisibility() {
        if (!trajectoryChart) return;

        // Toggle hidden status based on selection
        for (let i = 0; i < 5; i++) {
            if (selectedJointIndex === -1) {
                // Show All
                trajectoryChart.setDatasetVisibility(i, true);
            } else {
                // Show only selected
                trajectoryChart.setDatasetVisibility(i, i === selectedJointIndex);
            }
        }
        trajectoryChart.update();
    }

    // Call Init
    initChart();

    // --- 3D Visualization Logic (Three.js) ---
    // 3D Globals
    let camera, scene, renderer, controls;
    let robotParts = {};
    let targetBlock;
    let heldObject = null;

    function init3D() {
        console.log("Initializing 3D Scene...");
        const container = document.getElementById('three-container');
        if (!container) {
            console.error("3D Container not found!");
            return;
        }

        // Scene setup - Transparent Background
        // Scene setup - Transparent Background
        scene = new THREE.Scene();
        scene.background = null;

        // Camera
        // Ensure non-zero dimensions
        const width = container.clientWidth || 400;
        const height = container.clientHeight || 300;
        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(15, 12, 15); // Closer and more centered
        camera.lookAt(0, 4, 0); // Focus on robot mid-section (approx J2/J3 height)

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(window.devicePixelRatio); // Sharp rendering
        renderer.setSize(width, height, false); // false because we use CSS width: 100%
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); // Brighter
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(10, 30, 20);
        dirLight.castShadow = true;
        // Improve shadow quality
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.bias = -0.0001;
        scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x00f3ff, 0.8, 50);
        pointLight.position.set(-5, 10, -5);
        scene.add(pointLight);

        // Env Group
        const envGroup = new THREE.Group();
        scene.add(envGroup);
        robotParts.env = envGroup;

        // Default Floor (Hidden if toggle off)

        // Initial Grid (We will manage this in updateSceneObjects)
        // const gridHelper = new THREE.GridHelper(50, 50, 0x00f3ff, 0x333333);
        // scene.add(gridHelper);

        // Axes Helper
        scene.add(new THREE.AxesHelper(3));

        // Orbit Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 2; // Allow closer zoom
        controls.maxDistance = 150; // Allow further zoom out
        controls.maxPolarAngle = Math.PI / 2; // Don't go below ground
        controls.target.set(0, 4, 0); // Focus on robot mid-section

        // Enhanced Rviz-style Mouse Map
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN
        };
        // Also enable touch
        controls.touches = {
            ONE: THREE.MOUSE.ROTATE,
            TWO: THREE.MOUSE.DOLLY_PAN
        };

        // --- Target Object (Pick & Place) ---
        const boxGeo = new THREE.BoxGeometry(2, 2, 2);
        const boxMat = new THREE.MeshStandardMaterial({ color: 0xff9e00, roughness: 0.5, metalness: 0.5 });
        targetBlock = new THREE.Mesh(boxGeo, boxMat);
        targetBlock.castShadow = true;
        targetBlock.receiveShadow = true;

        resetTargetBlock();
        scene.add(targetBlock);

        // --- Build Robot (Industrial Style) ---
        // Materials
        const matBody = new THREE.MeshStandardMaterial({
            color: 0x8899aa, // Light grey/metallic
            roughness: 0.3,
            metalness: 0.8
        });
        const matDark = new THREE.MeshStandardMaterial({
            color: 0x222222,
            roughness: 0.7,
            metalness: 0.4
        });
        const matJoint = new THREE.MeshStandardMaterial({
            color: 0x00ff41, // Neon Green
            roughness: 0.5,
            metalness: 0.5,
            emissive: 0x003311,
            emissiveIntensity: 0.5
        });

        // --- Build Robot (Industrial Style | GLTF Support) ---
        const gltfLoader = new THREE.GLTFLoader();

        // Helper: Tries to load a GLB, falls back to the provided primitive
        function createRobotPart(partName, parentGroup, primitiveBuilder) {
            // 1. Create and Attach Primitive (Immediate Feedback)
            const primitiveMesh = primitiveBuilder();
            primitiveMesh.name = "viz_primitive";
            primitiveMesh.castShadow = true;
            primitiveMesh.receiveShadow = true;
            parentGroup.add(primitiveMesh);

            // 2. Attempt to Load Mesh
            const url = `assets/${partName}.glb`;
            gltfLoader.load(url,
                (gltf) => {
                    console.log(`Loaded mesh for ${partName}`);
                    const mesh = gltf.scene;

                    // Normalize?
                    // mesh.scale.set(10, 10, 10); // Adjust based on your model export units!

                    mesh.name = "viz_mesh";
                    mesh.castShadow = true;
                    mesh.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });

                    // Remove Primitive
                    const old = parentGroup.getObjectByName("viz_primitive");
                    if (old) parentGroup.remove(old);

                    parentGroup.add(mesh);
                },
                undefined, // Progress
                (err) => {
                    // console.warn(`No custom mesh found for ${partName}, keeping primitive.`);
                }
            );
        }

        // 1. Base Pedestal
        const robotRoot = new THREE.Group();
        scene.add(robotRoot);
        robotParts.root = robotRoot;

        // Base Static Group
        const baseGroup = new THREE.Group();
        robotRoot.add(baseGroup);

        createRobotPart('base', baseGroup, () => {
            const g = new THREE.Mesh(new THREE.CylinderGeometry(4, 5, 1, 32), matDark);
            g.position.y = 0.5;
            return g;
        });

        // 2. J1: Turret
        robotParts.j1 = new THREE.Group();
        robotParts.j1.position.y = 1;
        robotRoot.add(robotParts.j1);

        createRobotPart('j1', robotParts.j1, () => {
            const g = new THREE.Mesh(new THREE.CylinderGeometry(2, 2.5, 3, 32), matBody);
            g.position.y = 1.5;
            return g;
        });

        // 3. J2: Shoulder
        robotParts.j2 = new THREE.Group();
        robotParts.j2.position.y = 3; // Stack height
        robotParts.j1.add(robotParts.j2);

        // Visual for Link 1 (J2 -> J3)
        createRobotPart('j2_link', robotParts.j2, () => {
            const group = new THREE.Group();
            // Axle
            const axle = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 3.5, 16).rotateZ(Math.PI / 2), matDark);
            group.add(axle);
            // Arm
            const arm = new THREE.Mesh(new THREE.BoxGeometry(2, 8, 1.8), matBody);
            arm.position.y = 4;
            group.add(arm);
            return group;
        });

        // 4. J3: Elbow
        robotParts.j3 = new THREE.Group();
        robotParts.j3.position.y = 7; // Link 1 length approx
        // Note: The previous logic had nested groups inside groups acting as visual containers.
        // We attach J3 to J2 directly for kinematics.
        // But visually, the "Link 1" moves with J2.
        // The "Link 2" moves with J3.
        robotParts.j2.add(robotParts.j3);

        createRobotPart('j3_link', robotParts.j3, () => {
            const group = new THREE.Group();
            const axle = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 3, 16).rotateZ(Math.PI / 2), matDark);
            group.add(axle);
            const arm = new THREE.Mesh(new THREE.BoxGeometry(1.5, 6, 1.5), matBody);
            arm.position.y = 3;
            group.add(arm);
            return group;
        });

        // 5. J4: Wrist
        robotParts.j4 = new THREE.Group();
        robotParts.j4.position.y = 6;
        robotParts.j3.add(robotParts.j4);

        createRobotPart('j4_wrist', robotParts.j4, () => {
            const group = new THREE.Group();
            const cyl = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.5, 16), matDark);
            const hand = new THREE.Mesh(new THREE.BoxGeometry(1.2, 2, 0.5), matBody);
            hand.position.y = 1.5;
            group.add(cyl, hand);
            return group;
        });

        // 6. J5: Gripper (Claws)
        robotParts.j5Group = new THREE.Group();
        robotParts.j5Group.position.y = 2.5;
        robotParts.j4.add(robotParts.j5Group);

        // TCP Helper
        robotParts.tcp = new THREE.Object3D();
        robotParts.tcp.position.y = 1.0;
        robotParts.j5Group.add(robotParts.tcp);
        robotParts.tcp.add(new THREE.AxesHelper(1.5));

        // Gripper parts are distinct because they move (slide/rotate)
        // If we import a gripper, it might be a single static mesh, or two finger meshes.
        // For simplicity, we keep the procedural fingers unless 'gripper_l.glb' and 'gripper_r.glb' exist.

        robotParts.clawL = new THREE.Group();
        robotParts.clawL.position.set(-0.4, 0, 0);
        robotParts.j5Group.add(robotParts.clawL);

        createRobotPart('gripper_left', robotParts.clawL, () => {
            const m = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.5, 0.3), matJoint);
            m.position.y = 0.75;
            m.rotation.z = 0.2;
            return m;
        });

        robotParts.clawR = new THREE.Group();
        robotParts.clawR.position.set(0.4, 0, 0);
        robotParts.j5Group.add(robotParts.clawR);

        createRobotPart('gripper_right', robotParts.clawR, () => {
            const m = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.5, 0.3), matJoint);
            m.position.y = 0.75;
            m.rotation.z = -0.2;
            return m;
        });

        // Path Visualization Group
        robotParts.pathGroup = new THREE.Group();
        scene.add(robotParts.pathGroup);

        // --- VLA CAMERA (Global Overhead View) ---
        // Static camera positioned high up to capture the entire workspace (Extreme Reach)
        const vlaCamGroup = new THREE.Group();
        vlaCamGroup.position.set(0, 30, 0); // High overhead position
        vlaCamGroup.rotation.x = -Math.PI / 2; // Looking straight down

        // Attach to scene (Static) instead of robot arm
        scene.add(vlaCamGroup);

        const vlaCam = new THREE.PerspectiveCamera(60, 1, 0.1, 100); // Increased Far plane
        vlaCamGroup.add(vlaCam);

        const vlaCamHelper = new THREE.CameraHelper(vlaCam);
        scene.add(vlaCamHelper);

        // Add a visual body for the camera (Security Camera style)
        const camBodyGeo = new THREE.BoxGeometry(2, 1, 1);
        const camBodyMat = new THREE.MeshStandardMaterial({ color: 0x222222 });
        const camBody = new THREE.Mesh(camBodyGeo, camBodyMat);
        vlaCamGroup.add(camBody);

        // --- Spawn 10 Industrial Props (Robot Studio Style) ---
        window.spawnIndustrialProps = function (targetGroup) {
            console.log("Spawning industrial assets...");

            // Default to scene if not provided, though we prefer env group
            const groupToAdd = targetGroup || scene;

            const propsGroup = new THREE.Group();
            groupToAdd.add(propsGroup);

            // Helpers for materials
            const matWood = new THREE.MeshStandardMaterial({ color: 0x8b5a2b, roughness: 0.9 });
            const matMetal = new THREE.MeshStandardMaterial({ color: 0x777777, roughness: 0.4, metalness: 0.8 });
            const matYellow = new THREE.MeshStandardMaterial({ color: 0xffcc00, roughness: 0.5 });
            const matBlue = new THREE.MeshStandardMaterial({ color: 0x0055ff, roughness: 0.6 });
            const matRed = new THREE.MeshStandardMaterial({ color: 0xcc2200, roughness: 0.5 });
            const matDark = new THREE.MeshStandardMaterial({ color: 0x222222 });

            const assets = [
                // 1. Euro Pallet
                () => {
                    const g = new THREE.Group();
                    const plank = new THREE.BoxGeometry(2.4, 0.15, 0.4);
                    for (let i = 0; i < 3; i++) { // Top planks
                        const m = new THREE.Mesh(plank, matWood);
                        m.position.z = (i - 1) * 0.8;
                        m.position.y = 0.3;
                        g.add(m);
                    }
                    const runner = new THREE.BoxGeometry(0.2, 0.3, 2.0);
                    for (let i = 0; i < 3; i++) { // Runners
                        const m = new THREE.Mesh(runner, matWood);
                        m.position.x = (i - 1) * 1.0;
                        g.add(m);
                    }
                    return g;
                },
                // 2. Oil Barrel
                () => {
                    const g = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 1.8, 16), matBlue);
                    g.position.y = 0.9;
                    return g;
                },
                // 3. Shipping Crate
                () => {
                    const g = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.5, 1.5), matWood);
                    g.position.y = 0.75;
                    return g;
                },
                // 4. Safety Barrier (Yellow Fence)
                () => {
                    const g = new THREE.Group();
                    const poleGeo = new THREE.CylinderGeometry(0.1, 0.1, 2, 8);
                    const p1 = new THREE.Mesh(poleGeo, matYellow); p1.position.set(-1, 1, 0);
                    const p2 = new THREE.Mesh(poleGeo, matYellow); p2.position.set(1, 1, 0);
                    const barGeo = new THREE.BoxGeometry(2.2, 0.2, 0.1);
                    const b1 = new THREE.Mesh(barGeo, matDark); b1.position.set(0, 1.5, 0);
                    const b2 = new THREE.Mesh(barGeo, matDark); b2.position.set(0, 1.0, 0);
                    g.add(p1, p2, b1, b2);
                    return g;
                },
                // 5. Traffic Cone
                () => {
                    const g = new THREE.Group();
                    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.4, 1.2, 16), new THREE.MeshStandardMaterial({ color: 0xff6600 }));
                    cone.position.y = 0.6;
                    const base = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.1, 0.8), new THREE.MeshStandardMaterial({ color: 0x000000 }));
                    base.position.y = 0.05;
                    g.add(base, cone);
                    return g;
                },
                // 6. Steel Beam
                () => {
                    const g = new THREE.Group();
                    const w = 0.5, h = 3.0, d = 0.5;
                    const flange = new THREE.BoxGeometry(w, 0.1, h);
                    const web = new THREE.BoxGeometry(0.1, 0.5, h); // Rotated profile
                    const top = new THREE.Mesh(flange, matMetal); top.position.y = 0.25;
                    const bot = new THREE.Mesh(flange, matMetal); bot.position.y = -0.25;
                    const mid = new THREE.Mesh(web, matMetal);
                    g.add(top, bot, mid);
                    g.rotation.z = Math.PI / 2; // Lay flat
                    g.position.y = 0.3;
                    return g;
                },
                // 7. Parts Bin
                () => {
                    const g = new THREE.Group();
                    const box = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.5, 1.2), matRed);
                    box.position.y = 0.25;
                    // Hollow it out conceptually by adding a black top
                    const inner = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.1, 1.1), matDark);
                    inner.position.y = 0.5;
                    g.add(box, inner);
                    return g;
                },
                // 8. Concrete Block
                () => {
                    const g = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.8, 3), new THREE.MeshStandardMaterial({ color: 0x999999, roughness: 1 }));
                    g.position.y = 0.4;
                    return g;
                },
                // 9. E-Stop Stand
                () => {
                    const g = new THREE.Group();
                    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 2, 8), matMetal);
                    pole.position.y = 1;
                    const box = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.6, 0.3), matYellow);
                    box.position.y = 2;
                    const btn = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.05), matRed);
                    btn.position.set(0, 2, 0.15); btn.rotation.x = Math.PI / 2;
                    g.add(pole, box, btn);
                    return g;
                },
                // 10. Cable Drum
                () => {
                    const g = new THREE.Group();
                    const diskVal = new THREE.CylinderGeometry(0.8, 0.8, 0.1, 16);
                    const d1 = new THREE.Mesh(diskVal, matWood); d1.rotation.z = Math.PI / 2; d1.position.x = -0.4;
                    const d2 = new THREE.Mesh(diskVal, matWood); d2.rotation.z = Math.PI / 2; d2.position.x = 0.4;
                    const core = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.8, 16), matDark);
                    core.rotation.z = Math.PI / 2;
                    g.add(d1, d2, core);
                    g.position.y = 0.8;
                    return g;
                }
            ];

            // Spawn them in a semi-circle/random layout
            assets.forEach((createFn, i) => {
                const item = createFn();
                const angle = (i / assets.length) * Math.PI * 2;
                const radius = 10 + Math.random() * 5;

                item.position.x = Math.cos(angle) * radius;
                item.position.z = Math.sin(angle) * radius;

                // Random rotation
                item.rotation.y = Math.random() * Math.PI;

                item.castShadow = true;
                item.receiveShadow = true;
                item.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
                propsGroup.add(item);
            });
        };



        // --- Mouse Interaction (Drag Object) ---
        const raycaster = new THREE.Raycaster();
        const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -1); // Plane at y=1 (Block height)
        const pNormal = new THREE.Vector3(0, 1, 0);
        const shift = new THREE.Vector3();
        let isDraggingObj = false;

        container.addEventListener('mousedown', (e) => {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            if (targetBlock) {
                const intersects = raycaster.intersectObject(targetBlock);
                if (intersects.length > 0) {
                    controls.enabled = false; // Disable Orbit
                    isDraggingObj = true;
                    // Calculate shift
                    const intersectPoint = intersects[0].point;
                    shift.subVectors(targetBlock.position, intersectPoint);
                }
            }
        });

        container.addEventListener('mousemove', (e) => {
            if (!isDraggingObj) return;
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const targetPlane = new THREE.Plane(pNormal, -1); // Plane at y=1
            const intersectPoint = new THREE.Vector3();
            raycaster.ray.intersectPlane(targetPlane, intersectPoint);

            if (intersectPoint) {
                const newPos = intersectPoint.add(shift);
                // Boundary Check optional?
                targetBlock.position.x = newPos.x;
                targetBlock.position.z = newPos.z;
            }
        });

        container.addEventListener('mouseup', () => {
            if (isDraggingObj) {
                isDraggingObj = false;
                controls.enabled = true;
            }
        });

        // --- Advanced Scene Toggles Logic ---
        function setupAdvancedSceneToggles() {
            // 1. debug Grid
            const toggleGrid = document.getElementById('toggle-grid');
            let gridHelper = null;
            if (toggleGrid) {
                toggleGrid.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!gridHelper) gridHelper = new THREE.GridHelper(50, 50, 0x00f3ff, 0x333333);
                        scene.add(gridHelper);
                    } else {
                        if (gridHelper) scene.remove(gridHelper);
                    }
                });
            }

            // 2. World Axes
            const toggleAxes = document.getElementById('toggle-axes');
            let axesHelperGlobal = null;
            if (toggleAxes) {
                toggleAxes.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!axesHelperGlobal) axesHelperGlobal = new THREE.AxesHelper(5); // Bigger than local
                        scene.add(axesHelperGlobal);
                    } else {
                        if (axesHelperGlobal) scene.remove(axesHelperGlobal);
                    }
                });
            }

            // 3. VLA Cam View (CameraHelper)
            const toggleVLA = document.getElementById('toggle-vla-helper');
            // Assuming vlaCamHelper was created earlier in init3D.
            // We need to find reference. It wasn't stored in global scope.
            // Let's assume we can traverse or just recreate it? No, cleaner to store it.
            // Retrospective fix: Storing vlaCamHelper in robotParts or finding it.
            // For now, let's look it up by type if possible, or just rely on variable scope if this function is INSIDE init3D.
            // Yes, this code is being pasted INSIDE init3D. So local variables are accessible?
            // "vlaCamHelper" was defined in the previous edit block I made. 
            // So if I place this code *after* my previous edit, it should see "vlaCamHelper".
            if (toggleVLA && typeof vlaCamHelper !== 'undefined') {
                toggleVLA.addEventListener('change', (e) => {
                    vlaCamHelper.visible = e.target.checked;
                });
            }

            // 4. Target Block Visibility
            const toggleTarget = document.getElementById('toggle-target');
            if (toggleTarget) {
                toggleTarget.addEventListener('change', (e) => {
                    if (targetBlock) targetBlock.visible = e.target.checked;
                });
            }

            // 5. Safety Zone (Red Transparent Cylinder)
            const toggleZone = document.getElementById('toggle-zone');
            let safetyZone = null;
            if (toggleZone) {
                toggleZone.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!safetyZone) {
                            const geo = new THREE.CylinderGeometry(12, 12, 10, 32, 1, true);
                            const mat = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.1, side: THREE.DoubleSide });
                            safetyZone = new THREE.Mesh(geo, mat);
                            safetyZone.position.y = 5;
                        }
                        scene.add(safetyZone);
                    } else {
                        if (safetyZone) scene.remove(safetyZone);
                    }
                });
            }

            // 6. Conveyor Belt
            const toggleConveyor = document.getElementById('toggle-conveyor');
            let conveyorMesh = null;
            if (toggleConveyor) {
                toggleConveyor.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!conveyorMesh) {
                            const g = new THREE.Group();
                            // Belt
                            const belt = new THREE.Mesh(new THREE.BoxGeometry(4, 0.2, 20), new THREE.MeshStandardMaterial({ color: 0x111111 }));
                            // Rollers
                            const rollerGeo = new THREE.CylinderGeometry(0.5, 0.5, 4.2);
                            const r1 = new THREE.Mesh(rollerGeo, new THREE.MeshStandardMaterial({ color: 0x555555 }));
                            r1.rotation.z = Math.PI / 2; r1.position.z = -9;
                            const r2 = r1.clone(); r2.position.z = 9;
                            g.add(belt, r1, r2);
                            // Legs
                            const leg = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2, 0.2), new THREE.MeshStandardMaterial({ color: 0xaaaaaa }));
                            const l1 = leg.clone(); l1.position.set(1.9, -1, 9);
                            const l2 = leg.clone(); l2.position.set(-1.9, -1, 9);
                            const l3 = leg.clone(); l3.position.set(1.9, -1, -9);
                            const l4 = leg.clone(); l4.position.set(-1.9, -1, -9);
                            g.add(l1, l2, l3, l4);

                            g.position.set(-10, 2, 0); // Side of robot
                            conveyorMesh = g;
                        }
                        scene.add(conveyorMesh);
                    } else {
                        if (conveyorMesh) scene.remove(conveyorMesh);
                    }
                });
            }

            // 7. Ceiling Lights
            const toggleLights = document.getElementById('toggle-lights');
            let ceilingLightGroup = null;
            if (toggleLights) {
                toggleLights.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!ceilingLightGroup) {
                            ceilingLightGroup = new THREE.Group();
                            // Visual Fixtures
                            const fixGeo = new THREE.BoxGeometry(2, 0.2, 4);
                            const fixMat = new THREE.MeshStandardMaterial({ color: 0xdddddd, emissive: 0xaaaaaa });

                            const l1 = new THREE.Mesh(fixGeo, fixMat); l1.position.set(5, 15, 5);
                            const l2 = new THREE.Mesh(fixGeo, fixMat); l2.position.set(-5, 15, -5);
                            ceilingLightGroup.add(l1, l2);

                            // Actual RectAreaLights (expensive) or Spotlights
                            const spot1 = new THREE.SpotLight(0xffffff, 1);
                            spot1.position.set(5, 14, 5);
                            spot1.target.position.set(5, 0, 5);
                            const spot2 = new THREE.SpotLight(0xffffff, 1);
                            spot2.position.set(-5, 14, -5);
                            spot2.target.position.set(-5, 0, -5);
                            ceilingLightGroup.add(spot1, spot1.target, spot2, spot2.target);
                        }
                        scene.add(ceilingLightGroup);
                    } else {
                        if (ceilingLightGroup) scene.remove(ceilingLightGroup);
                    }
                });
            }

            // 8. Human Scale
            const toggleHuman = document.getElementById('toggle-human');
            let humanMesh = null;
            if (toggleHuman) {
                toggleHuman.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!humanMesh) {
                            const matSkin = new THREE.MeshStandardMaterial({ color: 0xffccaa });
                            const matShirt = new THREE.MeshStandardMaterial({ color: 0x0000ff });
                            const head = new THREE.Mesh(new THREE.SphereGeometry(0.3), matSkin); head.position.y = 1.7;
                            const body = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.8), matShirt); body.position.y = 1.2;
                            const legs = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.8), new THREE.MeshStandardMaterial({ color: 0x333333 })); legs.position.x = -0.15; legs.position.y = 0.4;
                            const legs2 = legs.clone(); legs2.position.x = 0.15;

                            humanMesh = new THREE.Group();
                            humanMesh.add(head, body, legs, legs2);
                            humanMesh.position.set(8, 0, 8); // Corner
                        }
                        scene.add(humanMesh);
                    } else {
                        if (humanMesh) scene.remove(humanMesh);
                    }
                });
            }

            // 9. Factory Walls
            const toggleFactory = document.getElementById('toggle-factory');
            let factoryWalls = null;
            if (toggleFactory) {
                toggleFactory.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!factoryWalls) {
                            const matWall = new THREE.MeshStandardMaterial({ color: 0x555555, side: THREE.BackSide, roughness: 0.8 });
                            // Giant box room
                            const geo = new THREE.BoxGeometry(60, 30, 60);
                            factoryWalls = new THREE.Mesh(geo, matWall);
                            factoryWalls.position.y = 14;
                            factoryWalls.receiveShadow = true;
                        }
                        scene.add(factoryWalls);
                    } else {
                        if (factoryWalls) scene.remove(factoryWalls);
                    }
                });
            }

            // 10. Fog
            const toggleFog = document.getElementById('toggle-fog');
            if (toggleFog) {
                toggleFog.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        scene.fog = new THREE.FogExp2(0x111111, 0.02);
                        // Update background to match fog if possible, but our bg is transparent/CSS
                        // Since scene.background is null, fog renders against transparent canvas which might look weird against black CSS body.
                        // Ideally we change scene background too.
                        scene.background = new THREE.Color(0x111111);
                    } else {
                        scene.fog = null;
                        scene.background = null; // Revert to transparent
                    }
                });
            }
            // 11. Safety Cage
            const toggleCage = document.getElementById('toggle-cage');
            let cageMesh = null;
            if (toggleCage) {
                toggleCage.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!cageMesh) {
                            cageMesh = new THREE.Group();
                            // Simple wire mesh fence
                            const postGeo = new THREE.BoxGeometry(0.5, 8, 0.5);
                            const postMat = new THREE.MeshStandardMaterial({ color: 0xffcc00 });
                            const meshMat = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, wireframe: true, transparent: true, opacity: 0.3 });

                            // 4 Corner posts
                            const p1 = new THREE.Mesh(postGeo, postMat); p1.position.set(-15, 4, -15);
                            const p2 = new THREE.Mesh(postGeo, postMat); p2.position.set(15, 4, -15);
                            const p3 = new THREE.Mesh(postGeo, postMat); p3.position.set(-15, 4, 15);
                            const p4 = new THREE.Mesh(postGeo, postMat); p4.position.set(15, 4, 15);

                            const wallGeo1 = new THREE.PlaneGeometry(30, 8);
                            const w1 = new THREE.Mesh(wallGeo1, meshMat); w1.position.set(0, 4, -15);
                            const w2 = new THREE.Mesh(wallGeo1, meshMat); w2.position.set(0, 4, 15); w2.rotation.y = Math.PI;
                            const w3 = new THREE.Mesh(wallGeo1, meshMat); w3.position.set(-15, 4, 0); w3.rotation.y = Math.PI / 2;
                            // Leave front open or gated? Open for now.

                            cageMesh.add(p1, p2, p3, p4, w1, w2, w3);
                        }
                        scene.add(cageMesh);
                    } else {
                        if (cageMesh) scene.remove(cageMesh);
                    }
                });
            }

            // 12. E-Stop Station
            const toggleEstop = document.getElementById('toggle-estop');
            let estopMesh = null;
            if (toggleEstop) {
                toggleEstop.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!estopMesh) {
                            estopMesh = new THREE.Group();
                            const box = new THREE.Mesh(new THREE.BoxGeometry(1.5, 2, 1.5), new THREE.MeshStandardMaterial({ color: 0xffff00 }));
                            const btn = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0xff0000 }));
                            btn.position.y = 1.0; btn.rotation.x = Math.PI / 2; // Facing up? No, on face?
                            // Let's make it a mushroom button on top
                            btn.rotation.x = 0; btn.position.y = 1.2;
                            estopMesh.add(box, btn);
                            estopMesh.position.set(5, 1, 10); // Front right
                        }
                        scene.add(estopMesh);
                    } else {
                        if (estopMesh) scene.remove(estopMesh);
                    }
                });
            }

            // 13. Control Laptop
            const toggleLaptop = document.getElementById('toggle-laptop');
            let laptopMesh = null;
            if (toggleLaptop) {
                toggleLaptop.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!laptopMesh) {
                            laptopMesh = new THREE.Group();
                            const base = new THREE.Mesh(new THREE.BoxGeometry(3, 0.2, 2.2), new THREE.MeshStandardMaterial({ color: 0x333333 }));
                            const screen = new THREE.Mesh(new THREE.BoxGeometry(3, 2, 0.1), new THREE.MeshStandardMaterial({ color: 0x333333 }));
                            screen.position.set(0, 1, -1.1);
                            screen.rotation.x = -0.2;

                            // Display
                            const disp = new THREE.Mesh(new THREE.PlaneGeometry(2.8, 1.8), new THREE.MeshBasicMaterial({ color: 0x00aaff }));
                            disp.position.set(0, 1, -1.04);
                            disp.rotation.x = -0.2;

                            laptopMesh.add(base, screen, disp);
                            laptopMesh.position.set(-5, 0.2, 8); // On table? Need to check table height. 
                            // Table is roughly at 0 if toggle-table is Off, or distinct if On.
                            // Assuming putting it on floor for now, or floating if table off.
                            laptopMesh.position.y = 10; // High in air? No, let's put on implicit table height
                        }
                        scene.add(laptopMesh);
                    } else {
                        if (laptopMesh) scene.remove(laptopMesh);
                    }
                });
            }

            // 14. Industrial Pipes
            const togglePipes = document.getElementById('toggle-pipes');
            let pipesMesh = null;
            if (togglePipes) {
                togglePipes.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!pipesMesh) {
                            pipesMesh = new THREE.Group();
                            const pipeMat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.3, metalness: 0.8 });
                            const p1 = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 40), pipeMat);
                            p1.rotation.z = Math.PI / 2; p1.position.set(0, 18, -18);
                            const p2 = p1.clone(); p2.position.set(0, 16, -18);
                            pipesMesh.add(p1, p2);
                        }
                        scene.add(pipesMesh);
                    } else {
                        if (pipesMesh) scene.remove(pipesMesh);
                    }
                });
            }

            // 15. Oil Drum
            const toggleDrum = document.getElementById('toggle-drum');
            let drumMesh = null;
            if (toggleDrum) {
                toggleDrum.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!drumMesh) {
                            const geo = new THREE.CylinderGeometry(1.5, 1.5, 4, 32);
                            const mat = new THREE.MeshStandardMaterial({ color: 0x3e2723 }); // Rusty Red/Brown
                            drumMesh = new THREE.Mesh(geo, mat);
                            drumMesh.position.set(-12, 2, -5);
                        }
                        scene.add(drumMesh);
                    } else {
                        if (drumMesh) scene.remove(drumMesh);
                    }
                });
            }

            // 16. Fire Extinguisher
            const toggleExt = document.getElementById('toggle-extinguisher');
            let extMesh = null;
            if (toggleExt) {
                toggleExt.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!extMesh) {
                            extMesh = new THREE.Group();
                            const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 2.5), new THREE.MeshStandardMaterial({ color: 0xff0000 }));
                            const nozzle = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 1), new THREE.MeshStandardMaterial({ color: 0x111111 }));
                            nozzle.position.set(0.3, 1, 0);
                            extMesh.add(cyl, nozzle);
                            extMesh.position.set(12, 1.25, 10);
                        }
                        scene.add(extMesh);
                    } else {
                        if (extMesh) scene.remove(extMesh);
                    }
                });
            }

            // 17. Overhead Monitor
            const toggleMonitor = document.getElementById('toggle-monitor');
            let monitorMesh = null;
            if (toggleMonitor) {
                toggleMonitor.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!monitorMesh) {
                            monitorMesh = new THREE.Group();
                            const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 5), new THREE.MeshStandardMaterial({ color: 0x333333 }));
                            arm.position.set(0, 15, 0); // High up
                            const screen = new THREE.Mesh(new THREE.BoxGeometry(6, 3.5, 0.2), new THREE.MeshStandardMaterial({ color: 0x111111 }));
                            screen.position.set(0, 12.5, 0);
                            screen.rotation.x = Math.PI / 6; // Tilt down

                            const content = new THREE.Mesh(new THREE.PlaneGeometry(5.8, 3.3), new THREE.MeshBasicMaterial({ color: 0x00ff00 }));
                            content.position.set(0, 12.5, 0.15);
                            content.rotation.x = Math.PI / 6;

                            monitorMesh.add(arm, screen, content);
                            monitorMesh.position.set(0, 0, -10);
                        }
                        scene.add(monitorMesh);
                    } else {
                        if (monitorMesh) scene.remove(monitorMesh);
                    }
                });
            }

            // 18. Warning Sign
            const toggleSign = document.getElementById('toggle-sign');
            let signMesh = null;
            if (toggleSign) {
                toggleSign.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!signMesh) {
                            signMesh = new THREE.Group();
                            const post = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 6), new THREE.MeshStandardMaterial({ color: 0x888888 }));
                            post.position.y = 3;
                            const board = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 0.1), new THREE.MeshStandardMaterial({ color: 0xffff00 }));
                            board.position.y = 5.5;
                            // Triangle shape text?
                            signMesh.add(post, board);
                            signMesh.position.set(-8, 0, 12);
                            signMesh.rotation.y = Math.PI / 4;
                        }
                        scene.add(signMesh);
                    } else {
                        if (signMesh) scene.remove(signMesh);
                    }
                });
            }

            // 19. Ventilation Duct
            const toggleVent = document.getElementById('toggle-vent');
            let ventMesh = null;
            if (toggleVent) {
                toggleVent.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!ventMesh) {
                            ventMesh = new THREE.Mesh(new THREE.BoxGeometry(50, 2, 4), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
                            ventMesh.position.set(0, 19, 5);
                        }
                        scene.add(ventMesh);
                    } else {
                        if (ventMesh) scene.remove(ventMesh);
                    }
                });
            }

            // 20. Sci-Fi Skybox
            const toggleScifi = document.getElementById('toggle-scifi');
            if (toggleScifi) {
                toggleScifi.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        scene.background = new THREE.Color(0x000033); // Deep Blue
                        scene.fog = new THREE.FogExp2(0x000033, 0.01);
                    } else {
                        scene.background = null; // Transparent
                        scene.fog = null;
                    }
                });
            }
        }
        setupAdvancedSceneToggles();

        // Initial Scene Update (After all components/functions are ready)
        updateSceneObjects();

        // Start render loop
        animate3D();
    }

    // --- Trajectory Visualization (Cylinder Mesh) ---
    function drawTrajectory(data) {
        if (!robotParts.pathGroup || !robotParts.tcp) return;

        // Ensure added to scene
        if (!robotParts.pathGroup.parent) scene.add(robotParts.pathGroup);

        // Clear old path
        while (robotParts.pathGroup.children.length > 0) {
            robotParts.pathGroup.remove(robotParts.pathGroup.children[0]);
        }

        const savedJoints = [...currentValues];
        let startJoints = [...currentValues];
        let prevPos = getTCPPosition(startJoints);

        // Colors & Config
        const COL_MOVEJ = 0xffff00; // Yellow for Curve
        const COL_MOVEL = 0x00ff00; // Green for Line
        const PATH_RADIUS = 0.15;   // Thick cylinder
        const NODE_RADIUS = 0.4;    // Sphere at waypoints

        // Helper to add sphere (Waypoint Node)
        const addNode = (pos) => {
            const geo = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
            const mat = new THREE.MeshBasicMaterial({ color: 0xff00ff, depthTest: false });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.copy(pos);
            mesh.renderOrder = 999;
            robotParts.pathGroup.add(mesh);
        };

        // Helper to add cylinder (Path Segment)
        const addSegment = (p1, p2, color) => {
            const vec = new THREE.Vector3().subVectors(p2, p1);
            const len = vec.length();
            if (len < 0.001) return;

            const geo = new THREE.CylinderGeometry(PATH_RADIUS, PATH_RADIUS, len, 8);
            const mat = new THREE.MeshBasicMaterial({ color: color, depthTest: false }); // Always on top
            const mesh = new THREE.Mesh(geo, mat);

            // Postion at midpoint
            mesh.position.copy(p1).add(vec.multiplyScalar(0.5));
            // Orient Y axis to vector
            // Reset vec because multiplyScalar modified it
            const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
            mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            mesh.renderOrder = 998;

            robotParts.pathGroup.add(mesh);
        };

        // Draw Start Node
        addNode(prevPos);

        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            const targetJoints = point.sensors;
            const motionType = point.motion ? point.motion.type : 'MoveJ';

            const segmentPoints = [];
            segmentPoints.push(prevPos);

            // Generate Points
            const segments = 20;
            for (let j = 1; j <= segments; j++) {
                const t = j / segments;
                let stepJoints = [];
                if (motionType === 'MoveJ') {
                    stepJoints = startJoints.map((s, k) => s + (targetJoints[k] - s) * t);
                    segmentPoints.push(getTCPPosition(stepJoints));
                }
            }
            if (motionType === 'MoveL') {
                segmentPoints.push(getTCPPosition(targetJoints));
            }

            // Draw Mesh Segments
            const color = motionType === 'MoveJ' ? COL_MOVEJ : COL_MOVEL;
            for (let k = 0; k < segmentPoints.length - 1; k++) {
                addSegment(segmentPoints[k], segmentPoints[k + 1], color);
            }

            // Draw End Node
            const wpPos = segmentPoints[segmentPoints.length - 1];
            addNode(wpPos);

            startJoints = targetJoints;
            prevPos = wpPos;
        }

        updateRobotModel(savedJoints);
    }

    function getTCPPosition(joints) {
        updateRobotModel(joints);
        // Force update of all matrices in chain
        if (robotParts.j1) robotParts.j1.updateMatrixWorld(true);
        if (robotParts.j2) robotParts.j2.updateMatrixWorld(true);
        if (robotParts.j3) robotParts.j3.updateMatrixWorld(true);
        if (robotParts.j4) robotParts.j4.updateMatrixWorld(true);
        if (robotParts.j5Group) robotParts.j5Group.updateMatrixWorld(true);

        robotParts.tcp.updateWorldMatrix(true, false);
        const vec = new THREE.Vector3();
        robotParts.tcp.getWorldPosition(vec);
        return vec;
    }

    function updateRobotModel(values) {
        if (!robotParts.j1) return;
        const rad1 = (values[0] - 90) * (Math.PI / 180);
        robotParts.j1.rotation.y = -rad1;
        const rad2 = (values[1] - 90) * (Math.PI / 180);
        robotParts.j2.rotation.z = rad2;
        const rad3 = (values[2] - 90) * (Math.PI / 180);
        robotParts.j3.rotation.z = rad3;
        const rad4 = (values[3] - 90) * (Math.PI / 180);
        robotParts.j4.rotation.z = rad4;
        const gripVal = (values[4] / 180);
        if (robotParts.clawL && robotParts.clawR) {
            const angle = gripVal * (Math.PI / 3);
            robotParts.clawL.rotation.z = -angle;
            robotParts.clawR.rotation.z = angle;
        }
        // Force matrix update for entire chain?
        // Scene update handles it usually, but we need immediate.
        // Traverse down from J1?
        robotParts.j1.updateMatrixWorld(true);
    }

    // Start animation loop
    function resizeRenderer() {
        const container = document.getElementById('three-container');
        if (!container || !camera || !renderer) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        const pixelRatio = renderer.getPixelRatio();

        // Check if resize is needed (buffer size vs CSS size * ratio)
        if (renderer.domElement.width !== Math.floor(width * pixelRatio) ||
            renderer.domElement.height !== Math.floor(height * pixelRatio)) {
            renderer.setSize(width, height, false); // false prevents resizing style
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }
    }

    function animate3D() {
        requestAnimationFrame(animate3D);
        if (!renderer || !scene || !camera) return;

        // Auto-resize if container changes (smooth anim)
        resizeRenderer();

        if (robotParts.j1) {
            // Kinematic Mapping
            // J1: Yaw (0-180 -> -90 to 90)
            const rad1 = (currentValues[0] - 90) * (Math.PI / 180);
            robotParts.j1.rotation.y = -rad1;

            // J2: Shoulder (0-180 -> -90 to 90)
            const rad2 = (currentValues[1] - 90) * (Math.PI / 180);
            robotParts.j2.rotation.z = rad2;

            // J3: Elbow
            const rad3 = (currentValues[2] - 90) * (Math.PI / 180);
            robotParts.j3.rotation.z = rad3;

            // J4: Wrist
            const rad4 = (currentValues[3] - 90) * (Math.PI / 180);
            robotParts.j4.rotation.z = rad4;

            // J5: Gripper
            // 0 (Closed) -> 180 (Open)
            const gripVal = (currentValues[4] / 180);
            if (robotParts.clawL && robotParts.clawR) {
                // Rotate base of fingers
                const angle = gripVal * (Math.PI / 3);
                robotParts.clawL.rotation.z = -angle;
                robotParts.clawR.rotation.z = angle;
            }
        }

        renderer.render(scene, camera);
        if (controls) controls.update();
        updatePhysics();

        // Update TCP Overlay
        if (robotParts.tcp) {
            const vec = new THREE.Vector3();
            robotParts.tcp.getWorldPosition(vec);
            // Convert to meters (assuming 1 unit = 10cm or similar, but let's just show raw div 10 for scale)
            // Actually let's assume 1 unit = 1 cm, so div 100 for meters.
            // Based on previous FK, values were around ~0.4 coords. 
            // In ThreeJS scene, arms are length 8, 6. Total ~14-20 units.
            // If L1=0.4m in FK logic but 8 units in 3D, scale is 20.
            const scale = 20;
            if (tcpValX) tcpValX.textContent = (vec.x / scale).toFixed(2);
            if (tcpValY) tcpValY.textContent = (vec.y / scale).toFixed(2);
            if (tcpValZ) tcpValZ.textContent = (vec.z / scale).toFixed(2);
        }
    }

    function resetTargetBlock() {
        if (targetBlock) {
            targetBlock.position.set(10, 1, 10); // x, y, z (y=1 sits on floor)
            if (scene) scene.attach(targetBlock);
            heldObject = null;
        }
    }

    function updatePhysics() {
        if (!targetBlock || !robotParts.tcp) return;

        // Simple Gravity for Object
        if (!heldObject) {
            if (targetBlock.position.y > 1) {
                targetBlock.position.y -= 0.5; // Gravity Fall
                if (targetBlock.position.y < 1) targetBlock.position.y = 1;
            }
        }

        // Pick & Place Logic
        const tcpPos = new THREE.Vector3();
        robotParts.tcp.getWorldPosition(tcpPos);
        const objPos = new THREE.Vector3();
        targetBlock.getWorldPosition(objPos);

        const dist = tcpPos.distanceTo(objPos);

        // J5 is Gripper: 0 = Closed (90 in array? No 0 in UI is 0 array).
        // Check updateDisplay: values[4] -> 0-180.
        // Viz: gripVal = values[4]/180. 0 is Closed? 
        // Logic in animate: angle = gripVal * PI/3.
        // If val=0 => angle=0 => Closed? 
        // Let's assume 0 is Closed.
        const gripVal = currentValues[4];

        // visual collision feedback
        if (dist < 3.5) {
            targetBlock.material.emissive.setHex(0x330000); // Red tint when close
        } else {
            targetBlock.material.emissive.setHex(0x000000);
        }

        // Pick
        if (dist < 3.0 && gripVal < 40 && !heldObject) {
            // Attach to TCP
            // Use Three.js attach method to preserve world transform
            robotParts.tcp.attach(targetBlock);
            heldObject = targetBlock;
            targetBlock.material.color.setHex(0x00ff00); // Green when held
        }

        // Drop
        if (heldObject && gripVal > 60) {
            scene.attach(heldObject);
            heldObject = null;
            targetBlock.material.color.setHex(0xff9e00); // Orignal Orange
        }
    }

    // Handle Resize
    window.addEventListener('resize', () => {
        const container = document.getElementById('three-container');
        if (container && camera && renderer) {
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }
    });

    const resetObjBtn = document.getElementById('reset-obj-btn');
    if (resetObjBtn) {
        resetObjBtn.addEventListener('click', resetTargetBlock);
    }

    // Call init at the end (Safe from TDZ)
    // --- Dynamic Scene Management ---


    // --- Dynamic Scene Management ---
    function updateSceneObjects() {
        if (!robotParts.env) return;

        const showFloor = document.getElementById('toggle-floor').checked;
        const showTable = document.getElementById('toggle-table').checked;
        const showObstacles = document.getElementById('toggle-obstacles').checked;
        const useShadows = document.getElementById('toggle-shadows').checked;

        // Clear Environment Group
        while (robotParts.env.children.length > 0) {
            robotParts.env.remove(robotParts.env.children[0]);
        }

        // Shadows
        renderer.shadowMap.enabled = useShadows;

        // 1. Grid Helper
        // Standard Grid if Floor OFF, otherwise on top or hidden
        if (!showFloor && !showTable) {
            const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0xcccccc);
            robotParts.env.add(gridHelper);
        }

        // 2. Realistic Floor (Infinite Workspace)
        if (showFloor) {
            const planeGeo = new THREE.PlaneGeometry(200, 200);
            const planeMat = new THREE.MeshStandardMaterial({
                color: 0x5d4037, // Earthy Brown ("Ground Life" feel)
                roughness: 0.9,
                metalness: 0.0,
                side: THREE.DoubleSide
            });
            const plane = new THREE.Mesh(planeGeo, planeMat);
            plane.rotation.x = -Math.PI / 2;
            plane.receiveShadow = true;

            // Floor position logic
            if (showTable) {
                plane.position.y = -10;
            } else {
                plane.position.y = 0;
            }
            robotParts.env.add(plane);
        }

        // 3. Work Table (Robot at Corner)
        if (showTable) {
            const tableHeight = 10;
            const tableWidth = 30; // Wider
            const tableDepth = 20; // Deeper

            const tableGeo = new THREE.BoxGeometry(tableWidth, tableHeight, tableDepth);
            const tableMat = new THREE.MeshStandardMaterial({
                color: 0xeeeeee, // White/Light Grey
                roughness: 0.5,
                metalness: 0.1
            });

            const table = new THREE.Mesh(tableGeo, tableMat);
            // Robot is at (0,0,0).
            // Position Table so (0,0) is a corner.
            // Let's make (0,0) the near-left corner.
            // Center X = tableWidth/2. Center Z = tableDepth/2.
            // And Y is down by height/2 (top at 0)

            table.position.set(tableWidth / 2 - 2, -tableHeight / 2, tableDepth / 2 - 2);
            // -2 offset to not be exactly on the edge, but comfortably in corner

            table.castShadow = true;
            table.receiveShadow = true;
            robotParts.env.add(table);
        }

        // 4. Obstacles (Industrial)
        if (showObstacles) {
            if (window.spawnIndustrialProps) {
                window.spawnIndustrialProps(robotParts.env);
            }
        }
    }

    function updatePhysics() {
        if (!targetBlock || !robotParts.tcp) return;

        // Physics Constants
        const boxRadius = 1;
        const tableCheck = document.getElementById('toggle-table');
        const hasTable = tableCheck ? tableCheck.checked : false;
        const floorLevel = hasTable ? -10 : 0;

        // Gravity
        if (!heldObject) {
            targetBlock.position.y -= 0.5;
        }

        // Constraints
        const x = targetBlock.position.x;
        const z = targetBlock.position.z;
        let groundLimit = floorLevel + boxRadius;

        if (hasTable) {
            // Check if above table
            // Table: x[-2, 28], z[-2, 18], as derived previously
            const onTable = (x > -2 && x < 28 && z > -2 && z < 18);
            if (onTable) {
                groundLimit = 0 + boxRadius; // Table Top at y=0
            }
        }

        // Floor/Table Collision
        if (targetBlock.position.y < groundLimit) {
            targetBlock.position.y = groundLimit;
        }

        // Pick & Place Logic
        const tcpPos = new THREE.Vector3();
        robotParts.tcp.getWorldPosition(tcpPos);
        const objPos = new THREE.Vector3();
        targetBlock.getWorldPosition(objPos);

        const dist = tcpPos.distanceTo(objPos);
        const gripVal = currentValues[4];

        // visual collision feedback
        if (dist < 3.5) {
            targetBlock.material.emissive.setHex(0x330000);
        } else {
            targetBlock.material.emissive.setHex(0x000000);
        }

        // Pick
        if (dist < 3.0 && gripVal < 40 && !heldObject) {
            robotParts.tcp.attach(targetBlock);
            heldObject = targetBlock;
            targetBlock.material.color.setHex(0x00ff00);
        }

        // Drop
        if (heldObject && gripVal > 60) {
            scene.attach(heldObject);
            heldObject = null;
            targetBlock.material.color.setHex(0xff9e00);
        }
    }

    // Toggle Listeners
    const toggleWall = document.getElementById('toggle-wall');
    const toggleShelf = document.getElementById('toggle-shelf');

    if (toggleFloor) toggleFloor.addEventListener('change', updateSceneObjects);
    if (toggleTable) toggleTable.addEventListener('change', updateSceneObjects);
    if (toggleObstacles) toggleObstacles.addEventListener('change', updateSceneObjects);
    if (toggleShadows) toggleShadows.addEventListener('change', updateSceneObjects);
    if (toggleWall) toggleWall.addEventListener('change', updateSceneObjects);
    if (toggleShelf) toggleShelf.addEventListener('change', updateSceneObjects);

    init3D();

    // Define resizeRenderer function
    function resizeRenderer() {
        const container = document.getElementById('three-container');
        if (container && camera && renderer) {
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }
    }

    // Use ResizeObserver for robust layout updates
    const resizeObserver = new ResizeObserver(() => {
        resizeRenderer();
    });

    const container = document.getElementById('three-container');
    if (container) {
        resizeObserver.observe(container);
    }

});
