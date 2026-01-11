#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <WebSocketsServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// --- Configuration ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Servo Pins (Adjust as needed for your board)
const int SERVO_PINS[5] = {13, 12, 14, 27, 26}; 

// --- globals ---
Servo servos[5];
AsyncWebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81); // Port 81 for WebSocket

// Store current angles
int currentAngles[5] = {90, 90, 90, 90, 90};

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(num);
      Serial.printf("[%u] Connected from %d.%d.%d.%d url: %s\n", num, ip[0], ip[1], ip[2], ip[3], payload);
    } break;
    case WStype_TEXT:
      // Payload is the JSON string: {"servos":[90,90,90,90,90]}
      // Parse it efficiently
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (!error) {
        JsonArray arr = doc["servos"];
        if (arr) {
           for (int i=0; i<5; i++) {
             if (i < arr.size()) {
               int val = arr[i];
               // Constrain and Write
               val = constrain(val, 0, 180);
               if (val != currentAngles[i]) {
                 servos[i].write(val);
                 currentAngles[i] = val;
               }
             }
           }
        }
      } else {
        Serial.print("JSON Error: ");
        Serial.println(error.c_str());
      }
      break;
  }
}

void setup() {
  Serial.begin(115200);

  // Attach Servos
  for (int i=0; i<5; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(90); // Home position
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Start WebSocket
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);

  // Start Server (Optional, if you want to serve the page from here, but mostly for WS)
  server.begin();
}

void loop() {
  webSocket.loop();
}
