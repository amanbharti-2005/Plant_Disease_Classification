
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from services.cloud_database import (
    register_device,
    save_observation,
)
from services.cloud_storage import upload_detection_image

IMAGE_PATH = Path("test_image.jpg")

try:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            "Place test_image.jpg in the project root."
        )

    device = register_device()
    observation_id = str(uuid4())

    # Upload the image under this observation's ID.
    image_path = upload_detection_image(
        image_path=str(IMAGE_PATH),
        device_id=device["id"],
        observation_id=observation_id,
    )

    # Test metadata only; no fabricated GPS or sensor readings.
    record = {
        "id": observation_id,
        "device_id": device["id"],
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "gps_fix_valid": False,
        "crop_name": None,
        "plant_health": "test_record",
        "image_path": image_path,
        "environment_snapshot": {},
    }

    result = save_observation(record)

    print("Observation saved successfully!")
    print("Observation ID:", observation_id)
    print("Image path:", image_path)
    print("Saved record:", result.data)

except Exception as error:
    print("Test failed:", type(error).__name__, str(error))
