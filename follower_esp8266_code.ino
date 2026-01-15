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

// --- PARSER ---
void parseAndMove(String payload) {
  // Expected: "v1,v2,v3,v4,v5"
  int start = 0;
  int index = 0;
  
  for (int i = 0; i < payload.length(); i++) {
    if (payload.charAt(i) == ',' || i == payload.length() - 1) {
      if (index >= 5) break; // Safety
      
      // Extract substring
      int end = (i == payload.length() - 1) ? i + 1 : i;
      String valStr = payload.substring(start, end);
      start = i + 1;
      
      int val = valStr.toInt();
      val = constrain(val, 0, 180);
      
      // Changed Logic for Pots:
      // Always write if valid, relying on Leader smoothing
      servos[index].write(val);
      currentAngles[index] = val;
      
      index++;
    }
  }
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[WS] Disconnected!\n");
      break;
    case WStype_CONNECTED: {
      Serial.printf("[WS] Connected to Leader!\n");
    } break;
    case WStype_TEXT:
      // Direct CSV Parse for Speed
      String text = String((char*)payload);
      parseAndMove(text);
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
