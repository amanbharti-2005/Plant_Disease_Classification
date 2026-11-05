
import os
from dotenv import load_dotenv
from supabase import create_client



load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY in .env"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def test_connection():
    # Returns an empty list if the table exists but has no rows.
    result = (
        supabase.table("devices")
        .select("id, device_name")
        .limit(5)
        .execute()
    )
    return result.data

def register_device():
    device_name = "AgroVision-Rover-01"

    existing = (
        supabase.table("devices")
        .select("id, device_name")
        .eq("device_name", device_name)
        .limit(1)
        .execute()
    )

    if existing.data:
        return existing.data[0]

    result = (
        supabase.table("devices")
        .insert({
            "device_name": device_name,
            "firmware_version": "initial",
            "status": "online"
        })
        .execute()
    )

    return result.data[0]

def save_sensor_reading(
    device_id,
    sensor_type,
    value,
    unit,
    status="valid"
):
    record = {
        "device_id": device_id,
        "sensor_type": sensor_type,
        "value": value,
        "unit": unit,
        "status": status
    }

    return (
        supabase
        .table("sensor_readings")
        .insert(record)
        .execute()
    )


def save_observation(record: dict):
    return (
        supabase
        .table("observations")
        .insert(record)
        .execute()
    )


