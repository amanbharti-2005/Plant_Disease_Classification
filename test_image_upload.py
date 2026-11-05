
from pathlib import Path
from uuid import uuid4

from services.cloud_storage import upload_detection_image
from services.cloud_database import register_device

# Replace this with the path to an existing JPEG image.
IMAGE_PATH = Path("test_image.jpg")

try:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            "Put a JPEG image named test_image.jpg "
            "in the project root first."
        )

    device = register_device()
    observation_id = str(uuid4())

    storage_path = upload_detection_image(
        image_path=str(IMAGE_PATH),
        device_id=device["id"],
        observation_id=observation_id
    )

    print("Image upload successful!")
    print("Storage path:", storage_path)

except Exception as error:
    print("Image upload failed:", type(error).__name__, str(error))
