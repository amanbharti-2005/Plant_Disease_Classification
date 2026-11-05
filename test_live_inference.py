
import time

from services.live_monitor import LiveMonitor
from services.live_inference import LiveInferenceWorker


camera = LiveMonitor()
worker = LiveInferenceWorker(camera)

try:
    print("Starting camera...")
    camera.start()

    # Wait for the first camera frame.
    for _ in range(15):
        if camera.get_latest_frame() is not None:
            break
        time.sleep(1)

    if camera.get_latest_frame() is None:
        print("FAILED: No camera frame received.")
        print("Camera status:", camera.get_status())

    else:
        print("Camera connected:", camera.get_status())

        print("Starting AI inference...")
        worker.start()

        # Allow up to 60 seconds for model loading and inference.
        for attempt in range(60):
            result = worker.get_latest_result()

            if result is not None:
                print("\nSUCCESS: AI inference completed.")
                print("Worker status:", worker.get_status())
                print("Disease:", result.get("disease"))
                print("Species:", result.get("species"))
                print("Condition:", result.get("condition"))
                print("Confidence:", result.get("confidence"))
                print("Plant locked:", result.get("plant_locked"))

                annotated = result.get("labeled_frame")

                if annotated is not None:
                    import cv2
                    cv2.imwrite("test_live_ai_result.jpg", annotated)
                    print("Saved test_live_ai_result.jpg")

                break

            status = worker.get_status()

            if status.get("error"):
                print("Inference error:", status["error"])

            time.sleep(1)

        else:
            print("FAILED: No inference result within 60 seconds.")
            print("Worker status:", worker.get_status())

finally:
    worker.stop()
    camera.stop()
