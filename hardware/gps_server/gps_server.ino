/*
  GPS module firmware — now exposes an HTTP endpoint (/gps) over WiFi,
  same pattern as the ESP32-CAM's /capture endpoint. Your Python backend
  fetches this whenever a scan happens, instead of you reading numbers
  off the Serial Monitor by hand.

  Board: whatever ESP32 is wired to the GPS module (GPIO16=RX, GPIO17=TX,
  as in your original code). This can be a DIFFERENT physical board from
  the ESP32-CAM — they just both need to be on the same WiFi network.
*/
#include <WiFi.h>
#include <WebServer.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

// ---- Fill in your WiFi credentials ----
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

TinyGPSPlus gps;
HardwareSerial GPSSerial(2);
WebServer server(80);

// Continuously updated by loop() from the GPS stream, served on demand
// by handleGps() — same idea as the camera board keeping a live frame
// buffer ready for /capture requests.
double lastLat = 0.0;
double lastLng = 0.0;
bool hasFix = false;
int lastSatellites = 0;

void handleGps() {
  String json = "{";
  json += "\"valid\":" + String(hasFix ? "true" : "false") + ",";
  json += "\"lat\":" + String(lastLat, 6) + ",";
  json += "\"lng\":" + String(lastLng, 6) + ",";
  json += "\"satellites\":" + String(lastSatellites);
  json += "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  GPSSerial.begin(9600, SERIAL_8N1, 16, 17);

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("GPS module ready! Use this in Streamlit sidebar: ");
  Serial.println(WiFi.localIP());
  Serial.println("(test in a browser: http://<that-ip>/gps )");

  server.on("/gps", handleGps);
  server.begin();
}

void loop() {
  while (GPSSerial.available()) {
    gps.encode(GPSSerial.read());
  }

  if (gps.location.isValid()) {
    lastLat = gps.location.lat();
    lastLng = gps.location.lng();
    hasFix = true;
  } else {
    hasFix = false;
  }
  lastSatellites = gps.satellites.value();

  server.handleClient();  // must be called every loop, keeps /gps responsive
}
