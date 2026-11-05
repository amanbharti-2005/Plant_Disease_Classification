import os
from dotenv import load_dotenv

# ==========================================================
# ENVIRONMENT
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


# ==========================================================
# MODEL PATHS
# ==========================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")

DISEASE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "plant_disease_model.keras"
)

YOLO_GENERAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

LEAF_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "leaf_detector_best.pt"
)

PEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ip102_pest_classifier_best.keras"
)

PEST_CLASS_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "pest_class_names.json"
)


# ==========================================================
# MODEL AVAILABILITY
# ==========================================================

ENABLE_LEAF_DETECTION = os.path.exists(LEAF_MODEL_PATH)

ENABLE_PEST_CLASSIFICATION = os.path.exists(PEST_MODEL_PATH)


# ==========================================================
# YOLO DETECTION SETTINGS
# ==========================================================

PLANT_CLASS_ID = 58

PLANT_MIN_CONFIDENCE = 0.40

GENERAL_MIN_CONFIDENCE = 0.35


# ==========================================================
# DISEASE CLASSIFICATION SETTINGS
# ==========================================================

DISEASE_MIN_CONFIDENCE = 0.55

GREEN_RATIO_MIN = 0.12


# ==========================================================
# PEST CLASSIFICATION SETTINGS
# ==========================================================

PEST_MIN_CONFIDENCE = 0.50


# ==========================================================
# TEMPORARY FILES
# ==========================================================

TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================================
# DATABASE
# ==========================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "agrovision.db"
)


# ==========================================================
# GROQ AI CONFIGURATION
# ==========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ==========================================================
# FASTAPI
# ==========================================================

API_HOST = "127.0.0.1"

API_PORT = 8001


# ==========================================================
# ESP32-CAM CONFIGURATION
# ==========================================================

ESP32_CAM_IP = "192.168.1.37"

ESP32_STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

ESP32_CAPTURE_URL = f"http://{ESP32_CAM_IP}/capture"


# ==========================================================
# GPS CONFIGURATION
# ==========================================================

ENABLE_GPS_TRACKING = False

GPS_DEFAULT_IP = "192.168.1.50"


# ==========================================================
# LIVE MONITORING
# ==========================================================

ENABLE_LIVE_MONITORING = True

ENABLE_LIVE_INFERENCE = True

INFERENCE_INTERVAL_SECONDS = 1.0

NEGATIVE_OBSERVATION_INTERVAL_SECONDS = 30

EVENT_COOLDOWN_SECONDS = 30


# ==========================================================
# IMAGE PROCESSING
# ==========================================================

ENABLE_DIGITAL_ZOOM = True

DIGITAL_ZOOM_FACTOR = 1.5


# ==========================================================
# SUPABASE IMAGE STORAGE
# ==========================================================

ENABLE_SUPABASE_PERSISTENCE = True

SUPABASE_BUCKET_NAME = "disease-detection-images"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")

SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DB_PATH = os.path.join(BASE_DIR, "agrovision.db")