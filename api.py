
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pathlib import Path
import shutil
import os
import cv2
import base64

from predict import predict_disease_pipeline
from utils.class_names import CLASS_NAMES
from utils.llm_engine import get_diagnosis, answer_question
import database


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AgroVision Rover Pipeline API",
    version="6.0",
    description="Plant detection, disease classification and AI diagnosis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "temp"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

NO_RESULT_STATUSES = {
    "No Target Object Spotted",
    "Invalid Image",
    "Unrecognized / Low Confidence"
}


class AskRequest(BaseModel):
    species: str
    condition: str
    question: str


# ============================================================
# IMAGE CONVERSION
# ============================================================

def mat_to_base64(img_matrix):
    """
    Convert an OpenCV BGR image into a base64 JPEG.
    Returns None if no image is available.
    """

    if img_matrix is None:
        return None

    if not hasattr(img_matrix, "size") or img_matrix.size == 0:
        return None

    success, buffer = cv2.imencode(
        ".jpg",
        img_matrix,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        return None

    encoded = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def to_float_or_none(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_confidence(value):
    """
    The prediction pipeline returns confidence as a fraction:
    0.0 to 1.0.

    The frontend receives confidence as a percentage:
    0.0 to 100.0.
    """

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 0 <= confidence <= 1:
        confidence *= 100

    return round(confidence, 2)


# ============================================================
# RESPONSE BUILDER
# ============================================================

def build_response(result: dict, latitude=None, longitude=None):

    disease = result.get(
        "disease",
        "Unrecognized / Low Confidence"
    )

    species = result.get("species", "Unknown")
    condition = result.get("condition", disease)

    raw_confidence = result.get("confidence", 0.0)
    confidence_percent = normalize_confidence(raw_confidence)

    is_valid = (
        disease not in NO_RESULT_STATUSES
        and confidence_percent > 0
    )

    # --------------------------------------------------------
    # AI DIAGNOSIS
    # --------------------------------------------------------

    if is_valid:
        try:
            diagnosis = get_diagnosis(
                species,
                condition,
                float(raw_confidence)
            )

        except Exception as exc:
            print(f"[API] Diagnosis generation error: {exc}")

            diagnosis = {
                "severity": "UNKNOWN",
                "recommendations": [],
                "treatment": "AI diagnosis could not be generated."
            }

    else:
        diagnosis = {
            "severity": "N/A",
            "recommendations": [],
            "treatment": "No confident plant disease diagnosis available."
        }

    # --------------------------------------------------------
    # DATABASE STORAGE
    # --------------------------------------------------------

    report_id = None

    if is_valid:
        try:
            report_id = database.save_scan(
                species=species,
                condition=condition,
                severity=diagnosis.get("severity", "UNKNOWN"),
                confidence=float(raw_confidence),
                treatment=diagnosis.get("treatment", ""),
                recommendations=diagnosis.get("recommendations", []),
                thumbnail=mat_to_base64(
                    result.get("cropped_leaf")
                ),
                latitude=latitude,
                longitude=longitude,
                weed_count=0
            )

        except Exception as exc:
            print(f"[API] Database save error: {exc}")

    # --------------------------------------------------------
    # ALL ANNOTATED IMAGES
    # --------------------------------------------------------

    labeled_frame = mat_to_base64(
        result.get("labeled_frame")
    )

    all_objects_frame = mat_to_base64(
        result.get("all_objects_frame")
    )

    bounding_box_image = mat_to_base64(
        result.get("bounding_box_image")
    )

    cropped_leaf = mat_to_base64(
        result.get("cropped_leaf")
    )

    # Prefer the dedicated bounding-box image for display.
    annotated_image = (
        bounding_box_image
        or all_objects_frame
        or labeled_frame
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    response_data = {
        "success": disease != "Invalid Image",

        "report_id": report_id,

        # Disease results
        "disease": disease,
        "species": species,
        "condition": condition,
        "confidence": confidence_percent,

        # AI diagnosis
        "severity": diagnosis.get("severity", "N/A"),
        "recommendations": diagnosis.get("recommendations", []),
        "treatment": diagnosis.get("treatment", ""),

        # Plant detection
        "plant_locked": result.get("plant_locked", False),
        "leaf_count": result.get("leaf_count", 0),
        "detected_objects": result.get("detected_objects", []),

        # Pest classification
        "pest": result.get("pest"),
        "pest_confidence": normalize_confidence(
            result.get("pest_confidence", 0)
        ),
        "pest_status": result.get("pest_status"),
        "pest_model_available": result.get(
            "pest_model_available",
            False
        ),

        # Image outputs
        "labeled_frame": labeled_frame,
        "all_objects_frame": all_objects_frame,
        "bounding_box_image": bounding_box_image,
        "cropped_frame": cropped_leaf,

        # Main image for the frontend
        "annotated_image": annotated_image,

        # GPS
        "latitude": latitude,
        "longitude": longitude,

        "timestamp": datetime.now().isoformat()
    }

    return JSONResponse(content=response_data)


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AgroVision Rover API is operational"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api": "AgroVision Rover"
    }


@app.get("/model")
def model():
    return {
        "model": "Plant Disease Classification and Object Detection",
        "frameworks": [
            "TensorFlow",
            "Ultralytics YOLOv8",
            "Groq AI"
        ],
        "classes": len(CLASS_NAMES)
    }


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


# ============================================================
# IMAGE UPLOAD PREDICTION
# ============================================================

@app.post("/predict")
async def process_predict(
    file: UploadFile = File(...),
    latitude: Optional[str] = Form(None),
    longitude: Optional[str] = Form(None)
):

    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No image filename provided."
            }
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Only JPG, JPEG and PNG images are supported."
            }
        )

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + extension
    )

    file_path = UPLOAD_FOLDER / filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_disease_pipeline(str(file_path))

        return build_response(
            result,
            to_float_or_none(latitude),
            to_float_or_none(longitude)
        )

    except Exception as exc:
        print(f"[API] Upload prediction error: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc)
            }
        )

    finally:
        if file_path.exists():
            file_path.unlink()


# ============================================================
# ESP32-CAM PREDICTION
# ============================================================

@app.post("/predict_esp32")
async def process_esp32_predict(
    request: Request,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):

    image_bytes = await request.body()

    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Empty image received from ESP32-CAM."
            }
        )

    filename = (
        "esp32_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".jpg"
    )

    file_path = UPLOAD_FOLDER / filename

    try:
        with open(file_path, "wb") as file:
            file.write(image_bytes)

        result = predict_disease_pipeline(str(file_path))

        return build_response(
            result,
            latitude,
            longitude
        )

    except Exception as exc:
        print(f"[API] ESP32 prediction error: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc)
            }
        )

    finally:
        if file_path.exists():
            file_path.unlink()


# ============================================================
# AI FOLLOW-UP CHAT
# ============================================================

@app.post("/ask")
async def ask_question(req: AskRequest):

    try:
        answer = answer_question(
            req.species,
            req.condition,
            req.question
        )

        return {"answer": answer}

    except Exception as exc:
        print(f"[API] AI advisor error: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "answer": f"AI advisor error: {exc}"
            }
        )


# ============================================================
# REPORTS AND AFFECTED AREAS
# ============================================================

@app.get("/affected-areas")
def affected_areas():
    return {
        "areas": database.get_affected_areas()
    }


@app.get("/reports")
def reports(
    species: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None
):
    return {
        "reports": database.get_all_reports(
            species=species,
            severity=severity,
            status=status
        )
    }


@app.get("/reports/{report_id}")
def report_detail(report_id: int):

    report = database.get_report(report_id)

    if report is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Report not found"}
        )

    return report


@app.post("/reports/{report_id}/resolve")
def resolve_report(report_id: int):

    success = database.mark_resolved(report_id)

    if not success:
        return JSONResponse(
            status_code=404,
            content={"error": "Report not found"}
        )

    return {
        "success": True,
        "report_id": report_id,
        "status": "resolved"
    }


# ============================================================
# ABOUT
# ============================================================

@app.get("/about")
def about():
    return {
        "developer": "AgroVision AI",
        "version": "6.0",
        "frameworks": [
            "TensorFlow",
            "Ultralytics YOLOv8",
            "Groq AI",
            "SQLite"
        ]
    }
