
from pathlib import Path
from services.cloud_database import supabase


def upload_detection_image(
    image_path: str,
    device_id: str,
    observation_id: str
) -> str:
    file_path = Path(image_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {file_path}"
        )

    storage_path = (
        f"{device_id}/{observation_id}/{file_path.name}"
    )

    with file_path.open("rb") as image_file:
        supabase.storage.from_("disease-detection-images").upload(
            path=storage_path,
            file=image_file.read(),
            file_options={
                "content-type": "image/jpeg",
                "upsert": "true"
            }
        )

    return storage_path
