"""
Single source of truth for every tunable setting in the pipeline.

The whole point of this file: when you want to change a threshold, swap
a model file, or turn a feature on/off, you edit ONE line here — you
never touch predict.py / api.py, and you never need to retrain anything
just to change how confident YOLO needs to be, for example.

Everything downstream (predict.py, api.py, database.py) imports from
here instead of hardcoding values.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# VISION MODELS
# ============================================================
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.keras")

# General object detector — COCO-pretrained, ships with ultralytics.
# This is what powers the "here's everything I see" ticker.
YOLO_GENERAL_MODEL_PATH = "yolov8n.pt"

# Weed detector — a SEPARATE custom-trained YOLO model (not built yet).
# Once you train one in Colab (see WEED_TRAINING_NOTES.md), drop the
# .pt file at this path and weed detection turns on automatically —
# no code changes needed anywhere else in the project.
WEED_MODEL_PATH = os.path.join(BASE_DIR, "models", "weed_detector.pt")
ENABLE_WEED_DETECTION = os.path.exists(WEED_MODEL_PATH)

# ============================================================
# DETECTION THRESHOLDS — tune these freely, no retraining needed
# ============================================================
PLANT_CLASS_ID = 58              # COCO class id for "potted plant"
PLANT_MIN_CONFIDENCE = 0.40      # how sure YOLO must be to lock a plant target
GENERAL_MIN_CONFIDENCE = 0.35    # floor for the "all objects" ticker
WEED_MIN_CONFIDENCE = 0.35       # floor for weed detections (once weed model exists)
GREEN_RATIO_MIN = 0.12   # 12% se kam green pixels = "plant nahi hai" maan lo

# Disease classifier confidence floor — used by predict.py's confidence
# gate. Below this, the top prediction is NOT reported as a diagnosis;
# it's flagged as "Unrecognized / Low Confidence" instead (and skipped
# in the database / diagnosis pipeline via api.py's NO_RESULT_STATUSES).
# Raise this if you're still seeing too many wrong/forced guesses;
# lower it if too many legitimate scans are getting flagged as uncertain.
DISEASE_MIN_CONFIDENCE = 0.55

# ============================================================
# GPS TRACKING
# ============================================================
# Flip to True once the GPS ESP32 is back on the same network as this
# app. While False, the sidebar hides the GPS IP box entirely and scans
# skip trying to reach it — no 3-second network timeout on every scan,
# no dead UI element pointing at hardware that isn't reachable.
ENABLE_GPS_TRACKING = False

# ============================================================
# DATABASE
# ============================================================
DB_PATH = os.path.join(BASE_DIR, "agrovision.db")

# ============================================================
# LLM (Groq)
# ============================================================
# Switched from Gemini to Groq. Set the GROQ_API_KEY environment
# variable — get a free key at https://console.groq.com/keys
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"