/*************************************************
 * ESP32 – Leader Robot (1 Encoder + 4 Pots)
 * J2: Rotary Encoder (Pins 25, 26) with 0-180 Clamping
 * J1, J3, J4, J5: Potentiometers
 *************************************************/

#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

// ===== USER CONFIG =====
const char* ssid     = "Rittu";
const char* password = "flem1577";
// =======================

// ===== PIN DEFINITIONS =====
// J2: Rotary Encoder
#define J2_CLK 25
#define J2_DT  26

// Potentiometers (J1, J3, J4, J5)
// J1->32, J3->34, J4->35, J5->39
const int potPins[4] = {32, 34, 35, 39};

// Limits & Sensitivity
#define JOINT_MIN 0
#define JOINT_MAX 180
#define ENCODER_STEP 9  // 9 deg/click * 20 clicks/rev = 180 deg (1 Rev)

// ===== VARIABLES =====
volatile int j2_position = 90; // Start at 90
volatile int lastEncoded = 0;

// Smoothing for Pots
float smoothedRaw[4];
const float ALPHA = 0.85; // Faster response (Less smoothing)

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

// ===== INTERRUPT: ENCODER (J2) =====
void IRAM_ATTR updateEncoder() {
  int MSB = digitalRead(J2_CLK);
  int LSB = digitalRead(J2_DT);

  int encoded = (MSB << 1) | LSB; 
  int sum  = (lastEncoded << 2) | encoded; 

  // Standard Logic
  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) j2_position += ENCODER_STEP;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) j2_position -= ENCODER_STEP;

  lastEncoded = encoded; 

  // --- CLAMPING (0 to 180) ---
  // "Prevent rotation from 0 to 180" -> Hard limits
  if (j2_position < JOINT_MIN) j2_position = JOINT_MIN;
  if (j2_position > JOINT_MAX) j2_position = JOINT_MAX;
}

// ===== WebSocket Events =====
void onWebSocketEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("Client #%u connected\n", client->id());
  }
}

void setup() {
  Serial.begin(115200);

  // ---- Setup Encoder (J2) ----
  pinMode(J2_CLK, INPUT_PULLUP);
  pinMode(J2_DT, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(J2_CLK), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(J2_DT), updateEncoder, CHANGE);

  // ---- Setup Pots ----
  for(int i=0; i<4; i++) {
    smoothedRaw[i] = analogRead(potPins[i]);
  }

  // ---- WiFi ----
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());

  ws.onEvent(onWebSocketEvent);
  server.addHandler(&ws);
  server.begin();
}

void loop() {
  static unsigned long lastSend = 0;

  // Send at ~66 Hz (15ms)
  if (millis() - lastSend >= 15) {
    lastSend = millis();

    // 1. Get J2 (Encoder)
    // No mapping needed, direct degree control
    int val_j2 = j2_position;

    // 2. Get Pots (J1, J3, J4, J5)
    int currentPots[4];
    for (int i = 0; i < 4; i++) {
      int raw = analogRead(potPins[i]);
      smoothedRaw[i] = (ALPHA * raw) + ((1.0 - ALPHA) * smoothedRaw[i]);
      int val = map((int)smoothedRaw[i], 0, 4095, JOINT_MIN, JOINT_MAX);
      currentPots[i] = constrain(val, JOINT_MIN, JOINT_MAX);
    }

    // --- Message ---
    // Format: J1, J2, J3, J4, J5
    char msg[64];
    snprintf(msg, sizeof(msg), "%d,%d,%d,%d,%d",
             currentPots[0], // J1
             val_j2,         // J2 (Encoder)
             currentPots[1], // J3
             currentPots[2], // J4
             currentPots[3]  // J5
    );

    ws.textAll(msg);
    ws.cleanupClients();
  }
}
