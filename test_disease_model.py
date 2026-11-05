import numpy as np
import tensorflow as tf

from config import DISEASE_MODEL_PATH
from utils.preprocessing import preprocess_image
from utils.class_names import CLASS_NAMES

MODEL_PATH = DISEASE_MODEL_PATH

IMAGE_PATH = input(
    "Enter the full path of your test leaf image: "
).strip().strip('"')

print("\nLoading disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nModel input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Number of classes:", len(CLASS_NAMES))

print("\nPreprocessing image...")

image = preprocess_image(IMAGE_PATH)

print("Input shape:", image.shape)
print("Input minimum:", np.min(image))
print("Input maximum:", np.max(image))
print("Input dtype:", image.dtype)

print("\nRunning prediction...")

predictions = model.predict(image, verbose=0)[0]

print("\nOutput shape:", predictions.shape)
print("Prediction sum:", np.sum(predictions))

top_indices = np.argsort(predictions)[::-1][:5]

print("\n========== TOP 5 PREDICTIONS ==========")

for index in top_indices:

    class_name = (
        CLASS_NAMES[index]
        if index < len(CLASS_NAMES)
        else f"Unknown class {index}"
    )

    confidence = float(predictions[index]) * 100

    print(
        f"{class_name}: {confidence:.2f}%"
    )

print("\n========== VERIFICATION ==========")

if len(predictions) != len(CLASS_NAMES):

    print("ERROR: Model output count and class count do not match.")

elif not np.all(np.isfinite(predictions)):

    print("ERROR: Model produced invalid numerical predictions.")

else:

    print("Output count matches class count.")
    print("All prediction values are finite.")