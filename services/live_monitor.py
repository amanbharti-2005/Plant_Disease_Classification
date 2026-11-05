
import threading
import time

import cv2
import numpy as np
import requests


class LiveMonitor:
    """
    Capture JPEG snapshots from the ESP32-CAM /capture endpoint.
    The camera IP is supplied by the Streamlit sidebar.
    """

    def __init__(self, camera_ip="192.168.1.37"):
        self.camera_ip = camera_ip.strip()
        self.capture_url = f"http://{self.camera_ip}/capture"

        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

        self._latest_frame = None
        self._last_error = None
        self._running = False
        self._last_frame_time = None
        self._frames_received = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AgroVisionCamera",
        )

        self._thread.start()

    def stop(self):
        self._stop_event.set()

        thread = self._thread

        if (
            thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3)

    def _capture_loop(self):
        print(
            f"[CAMERA] Using snapshot polling: {self.capture_url}",
            flush=True,
        )

        with self._lock:
            self._running = True
            self._last_error = None

        try:
            while not self._stop_event.is_set():

                try:
                    response = requests.get(
                        self.capture_url,
                        timeout=(3, 8),
                    )

                    response.raise_for_status()

                    image_bytes = np.frombuffer(
                        response.content,
                        dtype=np.uint8,
                    )

                    frame = cv2.imdecode(
                        image_bytes,
                        cv2.IMREAD_COLOR,
                    )

                    if frame is None or frame.size == 0:
                        raise RuntimeError(
                            "ESP32 returned an invalid image."
                        )

                    with self._lock:
                        self._latest_frame = frame
                        self._last_error = None
                        self._last_frame_time = time.time()
                        self._frames_received += 1

                        received_count = self._frames_received

                    if received_count == 1:
                        print(
                            f"[CAMERA] First snapshot received: "
                            f"{frame.shape[1]}x{frame.shape[0]}",
                            flush=True,
                        )

                    self._stop_event.wait(0.5)

                except Exception as exc:

                    message = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    with self._lock:
                        self._last_error = message

                    print(
                        f"[CAMERA] Snapshot request failed: {message}",
                        flush=True,
                    )

                    self._stop_event.wait(1.0)

        finally:
            with self._lock:
                self._running = False

            print(
                "[CAMERA] Capture worker stopped.",
                flush=True,
            )

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    def get_status(self):
        with self._lock:
            return {
                "running": self._running,
                "has_frame": self._latest_frame is not None,
                "error": self._last_error,
                "last_frame_time": self._last_frame_time,
                "frames_received": self._frames_received,
            }
