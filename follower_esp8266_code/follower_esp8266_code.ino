#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>
#include <Servo.h>

// --- CONFIGURATION ---
const char* ssid     = "Rittu";
const char* password = "flem1577";
const char* leader_ip = "192.168.125.166"; // <--- ENTER LEADER VALUE HERE (Check Leader Serial)
const int   leader_port = 80;
const char* leader_path = "/ws";

// --- SERVO PINS (NodeMCU) ---
// D4, D5, D6, D7, D1
const int SERVO_PINS[5] = {D4, D5, D6, D7, D1};

// --- GLOBALS ---
Servo servos[5];
WebSocketsClient webSocket;
int currentAngles[5] = {90, 90, 90, 90, 90};

// --- BINARY PARSER ---
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[WS] Disconnected!\n");
      // Safety: Center Robots? Or Hold Last?
      break;
    case WStype_CONNECTED: {
      Serial.printf("[WS] Connected to Leader!\n");
    } break;
    
    // Handle Binary Data (7 Bytes)
    case WStype_BIN:
      if (length == 7 && payload[0] == 0xAA) {
        // [AA, J1, J2, J3, J4, J5, CS]
        uint8_t cs = (payload[1] + payload[2] + payload[3] + payload[4] + payload[5]) % 255;
        if (cs == payload[6]) {
          // Valid Packet
          for(int i=0; i<5; i++) {
             int angle = payload[i+1];
             angle = constrain(angle, 0, 180);
             servos[i].write(angle);
             currentAngles[i] = angle;
          }
        } 
      }
      break;
      
   case WStype_TEXT:
      // Ignored (or add legacy support if needed)
      break;
  }
}

void setup() {
  Serial.begin(115200);

  // Attach Servos
  for (int i = 0; i < 5; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(90);
  }

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  // WebSocket Client
  // Using Leader IP
  webSocket.begin(leader_ip, leader_port, leader_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(1000); 
}

void loop() {
  webSocket.loop();
}
