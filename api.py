from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import shutil
import os
import cv2
import base64

from predict import predict_disease_pipeline
from utils.class_names import CLASS_NAMES
from utils.llm_engine import get_diagnosis, answer_question
import database

app = FastAPI(
    title="AgroVision Rover Pipeline API",
    version="5.0",
    description="YOLO + MobileNet + Gemini LLM + persistent geotagged disease memory"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NO_RESULT_STATUSES = {"No Target Object Spotted", "Invalid Image", "Unrecognized / Low Confidence"}


class AskRequest(BaseModel):
    species: str
    condition: str
    question: str


def mat_to_base64(img_matrix):
    if img_matrix is None:
        return None
    ok, buffer = cv2.imencode('.jpg', img_matrix)
    if not ok:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def _to_float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_response(result: dict, latitude=None, longitude=None):
    disease = result["disease"]
    confidence = result["confidence"]
    is_valid = disease not in NO_RESULT_STATUSES
    species = result.get("species", "Unknown")
    condition = result.get("condition", disease)
    weed_count = result.get("weed_count", 0)

    if is_valid:
        diagnosis = get_diagnosis(species, condition, confidence)
    else:
        diagnosis = {
            "severity": "CLEAR",
            "recommendations": [],
            "treatment": "No targets detected in path.",
        }

    report_id = None
    if is_valid:
        # This is the actual "memory" — every real detection (disease OR
        # healthy) gets a permanent, geotagged row. Only skipped for
        # "No Target"/"Invalid Image" since there's nothing worth
        # remembering there.
        report_id = database.save_scan(
            species=species,
            condition=condition,
            severity=diagnosis["severity"],
            confidence=confidence,
            treatment=diagnosis["treatment"],
            recommendations=diagnosis["recommendations"],
            thumbnail=mat_to_base64(result.get("cropped_leaf")),
            latitude=latitude,
            longitude=longitude,
            weed_count=weed_count,
        )

    return JSONResponse({
        "success": disease != "Invalid Image",
        "report_id": report_id,
        "disease": disease,
        "species": species,
        "condition": condition,
        "confidence": round(confidence * 100, 2) if is_valid else 0.0,
        "severity": diagnosis["severity"],
        "recommendations": diagnosis["recommendations"],
        "treatment": diagnosis["treatment"],
        "weed_count": weed_count,
        "latitude": latitude,
        "longitude": longitude,
        "labeled_frame": mat_to_base64(result.get("labeled_frame")),
        "cropped_frame": mat_to_base64(result.get("cropped_leaf")),
        "all_objects_frame": mat_to_base64(result.get("all_objects_frame")),
        "detected_objects": result.get("detected_objects", []),
        "timestamp": datetime.now().isoformat()
    })


@app.get("/")
def home():
    return {"status": "running", "message": "AgroVision Rover HUD Core Engine Operations"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model")
def model():
    return {
        "model": "Compound Object Detection & Plant Disease Classification",
        "frameworks": ["TensorFlow (MobileNetV2)", "Ultralytics (YOLOv8)", "Gemini"],
        "classes": len(CLASS_NAMES)
    }


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def process_predict(
    file: UploadFile = File(...),
    latitude: Optional[str] = Form(None),
    longitude: Optional[str] = Form(None),
):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = predict_disease_pipeline(file_path)
        return build_response(result, _to_float_or_none(latitude), _to_float_or_none(longitude))
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/predict_esp32")
async def process_esp32_predict(
    request: Request,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
):
    """lat/lon passed as query params for this raw-bytes endpoint,
    e.g. POST /predict_esp32?latitude=12.97&longitude=77.59"""
    image_bytes = await request.body()
    filename = "esp32_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    try:
        result = predict_disease_pipeline(file_path)
        return build_response(result, latitude, longitude)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        answer = answer_question(req.species, req.condition, req.question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error: {e}"})


# ============================================================
# MEMORY / MAP / REPORTS — this is the new part completing the loop
# ============================================================
@app.get("/affected-areas")
def affected_areas():
    """The rover's navigation code polls this to know where to go back to."""
    return {"areas": database.get_affected_areas()}


@app.get("/reports")
def reports(species: Optional[str] = None, severity: Optional[str] = None, status: Optional[str] = None):
    return {"reports": database.get_all_reports(species=species, severity=severity, status=status)}


@app.get("/reports/{report_id}")
def report_detail(report_id: int):
    report = database.get_report(report_id)
    if report is None:
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return report


@app.post("/reports/{report_id}/resolve")
def resolve_report(report_id: int):
    ok = database.mark_resolved(report_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return {"success": True, "report_id": report_id, "status": "resolved"}


@app.get("/weed-summary")
def weed_summary():
    """Answers 'which crop had how many weeds' across all scans."""
    return {"summary": database.get_weed_summary()}


@app.get("/about")
def about():
    return {
        "developer": "AgroVision AI",
        "version": "5.0",
        "frameworks": ["TensorFlow", "Ultralytics YOLOv8", "Google Gemini", "SQLite"],
    }