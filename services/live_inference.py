
import threading
import time
import traceback

import config
from predict import predict_frame_pipeline


class LiveInferenceWorker:
    """Run live inference on camera frames in a background thread."""

    def __init__(self, camera_monitor):
        self.camera_monitor = camera_monitor

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        self._latest_result = None
        self._last_inference_time = None
        self._last_error = None
        self._running = False
        self._inference_started_at = None

    def start(self):
        """Start a single inference worker."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="AgroVisionInference",
        )
        self._thread.start()

    def stop(self):
        """Request a graceful stop."""
        self._stop_event.set()

    def _inference_loop(self):
        interval = max(
            0.1,
            float(getattr(config, "INFERENCE_INTERVAL_SECONDS", 1.0)),
        )

        with self._lock:
            self._running = True
            self._last_error = None

        print("[LIVE AI] Worker started.", flush=True)

        try:
            while not self._stop_event.is_set():
                started_at = time.perf_counter()

                frame = self.camera_monitor.get_latest_frame()

                if frame is None:
                    self._stop_event.wait(0.2)
                    continue

                with self._lock:
                    self._inference_started_at = time.time()

                print("[LIVE AI] Inference started.", flush=True)

                try:
                    result = predict_frame_pipeline(frame)

                    if result is None:
                        raise RuntimeError(
                            "Prediction pipeline returned no result."
                        )

                    with self._lock:
                        self._latest_result = result
                        self._last_inference_time = time.time()
                        self._last_error = None

                    print("[LIVE AI] Inference completed.", flush=True)

                except Exception as exc:
                    error_message = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    with self._lock:
                        self._last_error = error_message

                    print(
                        f"[LIVE AI] Inference failed: {error_message}",
                        flush=True,
                    )
                    traceback.print_exc()

                elapsed = time.perf_counter() - started_at
                remaining = max(0.0, interval - elapsed)
                self._stop_event.wait(remaining)

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            print(
                f"[LIVE AI] Worker failed: {exc}",
                flush=True,
            )
            traceback.print_exc()

        finally:
            with self._lock:
                self._running = False

            print("[LIVE AI] Worker stopped.", flush=True)

    def get_latest_result(self):
        """Return the latest completed inference result."""
        with self._lock:
            return self._latest_result

    def get_status(self):
        """Return a snapshot of worker status."""
        with self._lock:
            return {
                "running": self._running,
                "has_result": self._latest_result is not None,
                "last_inference_time": self._last_inference_time,
                "inference_started_at": self._inference_started_at,
                "error": self._last_error,
                "thread_alive": (
                    self._thread is not None
                    and self._thread.is_alive()
                ),
            }
