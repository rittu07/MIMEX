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

// ===== SAVITZKY-GOLAY FILTER CLASS =====
// Start-point smoothing (Window=5, Poly=2)
// Coeffs: [3, -5, -3, 9, 31] / 35
class SavitzkyGolayFilter {
  private:
    float buffer[5] = {0};
    int head = 0;
    
  public:
    void init(float val) {
      for(int i=0; i<5; i++) buffer[i] = val;
    }

    float update(float startVal) {
      // Shift buffer (Slow but clear)
      for(int i=0; i<4; i++) buffer[i] = buffer[i+1];
      buffer[4] = startVal;

      float result = (3.0*buffer[0] - 5.0*buffer[1] - 3.0*buffer[2] + 9.0*buffer[3] + 31.0*buffer[4]) / 35.0;
      return result;
    }
};

SavitzkyGolayFilter filters[4];

// ===== SOFT LIMIT =====
// Decelerates motion as it approaches 0 or 180
int softLimit(int val) {
  const int MARGIN = 15; // Slow down within 15 degrees of edge
  if (val > 180 - MARGIN) {
    // Dampen the approach to 180
    // If val is 170 -> 170
    // If val is 179 -> 176 (Example)
    float delta = 180 - val;
    float damped = 180 - (delta + (180 - delta) * 0.05); // Simple resistance
    return min(val, 180); // Just clamp for now, advanced math needs velocity context
  }
  if (val < MARGIN) {
    return max(val, 0);
  }
  return val;
}

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

  // ---- Setup ADC Resolution ----
  // ESP32 default is 12-bit (0-4095)
  // analogReadResolution(12); 

  // ---- Initialize Filters ----
  for(int i=0; i<4; i++) {
     filters[i].init(analogRead(potPins[i]));
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

    int val_j2 = j2_position; // Encoder doesn't need smooth/ADC filter

    // Process ADC Pots (J1, J3, J4, J5)
    uint8_t currentPots[5]; // Store as Byte for Binary Pack
    
    // Encoder is J2 (Index 1)
    currentPots[1] = (uint8_t)constrain(val_j2, 0, 180);

    for (int i = 0; i < 4; i++) {
        // Map i (0,1,2,3) to Joint Index (0,2,3,4) -> J1, J3, J4, J5
        int jointIdx = (i == 0) ? 0 : (i + 1);

        int raw = analogRead(potPins[i]); // Simple read, S-G filter handles noise well
        
        float filtered = filters[i].update(raw);
        
        // Map Float
        int mapped = (int)(filtered * (float)(JOINT_MAX - JOINT_MIN) / 4095.0 + JOINT_MIN);
        
        // Soft Limit
        mapped = softLimit(mapped);
        
        currentPots[jointIdx] = (uint8_t)constrain(mapped, 0, 180);
    }

    // --- Message: Binary Protocol ---
    // [0xAA, J1, J2, J3, J4, J5, Checksum]
    // 7 Bytes total vs ~16 Bytes CSV
    uint8_t packet[7];
    packet[0] = 0xAA; // Header
    packet[1] = currentPots[0];
    packet[2] = currentPots[1];
    packet[3] = currentPots[2];
    packet[4] = currentPots[3];
    packet[5] = currentPots[4];
    packet[6] = (packet[1]+packet[2]+packet[3]+packet[4]+packet[5]) % 255; // Checksum

    ws.binaryAll(packet, 7);
    ws.cleanupClients();
  }
}
