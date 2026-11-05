
import time
import cv2
from services.live_monitor import LiveMonitor

monitor = LiveMonitor()
monitor.start()

try:
    for attempt in range(10):
        time.sleep(1)

        status = monitor.get_status()
        frame = monitor.get_latest_frame()

        print(f"Attempt {attempt + 1}: {status}")

        if frame is not None:
            print("SUCCESS: Camera frame received.")
            print("Frame dimensions:", frame.shape)

            cv2.imwrite("test_live_frame.jpg", frame)
            print("Saved test_live_frame.jpg")
            break
    else:
        print("FAILED: No camera frame received.")
        print("Check the ESP32-CAM stream URL and network.")

finally:
    monitor.stop()
