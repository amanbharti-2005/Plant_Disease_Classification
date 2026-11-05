
import cv2
from predict import predict_frame_pipeline

image = cv2.imread("test_live_frame.jpg")

if image is None:
    print("FAILED: test_live_frame.jpg could not be loaded.")
else:
    print("Input frame:", image.shape)

    result = predict_frame_pipeline(image)

    print("\nPrediction test completed.")
    print("Disease:", result.get("disease"))
    print("Species:", result.get("species"))
    print("Condition:", result.get("condition"))
    print("Confidence:", result.get("confidence"))
    print("Weed count:", result.get("weed_count"))
    print("Plant locked:", result.get("plant_locked"))
    print("Low confidence:", result.get("low_confidence"))

    annotated = result.get("labeled_frame")

    if annotated is not None:
        cv2.imwrite("test_prediction_result.jpg", annotated)
        print("Saved test_prediction_result.jpg")
    else:
        print("No annotated frame was returned.")
