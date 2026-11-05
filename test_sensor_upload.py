
from services.cloud_database import (
    register_device,
    save_sensor_reading
)

try:
    # Retrieve the existing rover instead of creating another.
    device = register_device()
    device_id = device["id"]

    # Clearly marked test record, not a real sensor measurement.
    result = save_sensor_reading(
        device_id=device_id,
        sensor_type="connection_test",
        value=1,
        unit="test",
        status="test"
    )

    print("Sensor upload successful!")
    print("Rover:", device["device_name"])
    print("Reading:", result.data)

except Exception as error:
    print("Upload failed:", type(error).__name__, str(error))
