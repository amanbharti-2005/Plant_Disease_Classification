import os
import tempfile
import json

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

import config
from utils.preprocessing import preprocess_image
from utils.class_names import CLASS_NAMES


# ==========================================================
# PATHS AND SETTINGS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)

TEMP_CROP_PATH = os.path.join(TEMP_DIR, "current_crop.jpg")

DISEASE_MIN_CONFIDENCE = float(config.DISEASE_MIN_CONFIDENCE)
GENERAL_MIN_CONFIDENCE = float(config.GENERAL_MIN_CONFIDENCE)
PLANT_MIN_CONFIDENCE = float(config.PLANT_MIN_CONFIDENCE)
GREEN_RATIO_MIN = float(config.GREEN_RATIO_MIN)

UNRECOGNIZED_LABEL = "Unrecognized / Low Confidence"


# ==========================================================
# LOAD MODELS
# ==========================================================

print("[predict] Loading disease classifier...")
disease_model = tf.keras.models.load_model(
    config.DISEASE_MODEL_PATH
)

print("[predict] Loading general YOLO...")
yolo_model = YOLO(config.YOLO_GENERAL_MODEL_PATH)

COCO_NAMES = yolo_model.names


# Dedicated leaf detector
leaf_model = None

if config.ENABLE_LEAF_DETECTION:
    print("[predict] Loading dedicated leaf detector...")
    leaf_model = YOLO(config.LEAF_MODEL_PATH)
else:
    print("[predict] Leaf detector not found.")


# Pest classifier
pest_model = None
pest_class_names = []

if config.ENABLE_PEST_CLASSIFICATION:
    print("[predict] Loading IP102 pest classifier...")
    pest_model = tf.keras.models.load_model(
        config.PEST_MODEL_PATH
    )

    if os.path.exists(config.PEST_CLASS_NAMES_PATH):
        try:
            with open(
                config.PEST_CLASS_NAMES_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                pest_class_names = json.load(file)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[predict] Pest class names error: {exc}")

print("[predict] Disease classifier: loaded")
print("[predict] General YOLO: loaded")
print("[predict] Leaf detector:", "loaded" if leaf_model else "unavailable")
print("[predict] Pest classifier:", "loaded" if pest_model else "unavailable")
print("[predict] Weed detection: disabled")


# ==========================================================
# SPECIES AND CONDITION PARSING
# ==========================================================

def split_species_condition(disease_name):

    name = str(disease_name)

    for separator in ("___", "__", "-"):

        if separator in name:

            species_raw, condition_raw = name.split(
                separator,
                1
            )

            species = species_raw.replace("_", " ").strip()
            condition = condition_raw.replace("_", " ").strip()

            return species, condition

    if "_" in name:

        species_raw, condition_raw = name.split("_", 1)

        return (
            species_raw.replace("_", " ").strip(),
            condition_raw.replace("_", " ").strip()
        )

    return "Unspecified Species", name.replace("_", " ")


# ==========================================================
# BOUNDING BOX VALIDATION
# ==========================================================

def valid_box(box, width, height):

    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ==========================================================
# DRAW DIAGNOSTIC LABEL
# ==========================================================

def draw_tech_borders(img, x1, y1, x2, y2, label):

    out = img.copy()

    h, w = out.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))

    x2 = max(x1 + 1, min(int(x2), w - 1))
    y2 = max(y1 + 1, min(int(y2), h - 1))

    color = (0, 255, 0)

    cv2.rectangle(
        out,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label_text = str(label).upper()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        label_text,
        font,
        font_scale,
        thickness
    )

    text_x = x1
    text_y = max(text_h + 10, y1 - 10)

    padding = 5

    cv2.rectangle(
        out,
        (text_x, max(0, text_y - text_h - padding)),
        (
            min(w - 1, text_x + text_w + padding * 2),
            min(h - 1, text_y + baseline + padding)
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        out,
        label_text,
        (text_x + padding, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

    return out


# ==========================================================
# DRAW ALL DETECTED OBJECTS
# ==========================================================

def draw_all_detections(img, boxes):

    out = img.copy()

    for det in boxes:

        x1, y1, x2, y2 = det["bbox"]

        color = (
            (0, 255, 0)
            if det.get("is_leaf")
            else (255, 200, 60)
        )

        label = (
            f'{det["label"]} '
            f'{int(det["confidence"] * 100)}%'
        )

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    return out


# ==========================================================
# GENERAL OBJECT DETECTION
# ==========================================================

def detect_general_objects(frame):

    results = yolo_model(
        frame,
        verbose=False
    )[0]

    h, w = frame.shape[:2]

    detected = []
    plant_bbox = None
    best_plant_confidence = 0.0

    for box in results.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence < GENERAL_MIN_CONFIDENCE:
            continue

        bbox = valid_box(
            box.xyxy[0].cpu().numpy(),
            w,
            h
        )

        if bbox is None:
            continue

        if isinstance(COCO_NAMES, dict):
            label = COCO_NAMES.get(
                class_id,
                f"class_{class_id}"
            )
        else:
            label = str(class_id)

        detected.append({
            "bbox": bbox,
            "label": str(label),
            "confidence": confidence,
            "is_leaf": False
        })

        if (
            class_id == config.PLANT_CLASS_ID
            and confidence >= PLANT_MIN_CONFIDENCE
            and confidence > best_plant_confidence
        ):

            plant_bbox = bbox
            best_plant_confidence = confidence

    return detected, plant_bbox


# ==========================================================
# DEDICATED LEAF DETECTION
# ==========================================================

def detect_leaves(frame):

    if leaf_model is None:
        return []

    results = leaf_model(
        frame,
        verbose=False
    )[0]

    h, w = frame.shape[:2]

    leaf_boxes = []
    names = leaf_model.names

    for box in results.boxes:

        confidence = float(box.conf[0])

        if confidence < PLANT_MIN_CONFIDENCE:
            continue

        bbox = valid_box(
            box.xyxy[0].cpu().numpy(),
            w,
            h
        )

        if bbox is None:
            continue

        class_id = int(box.cls[0])

        if isinstance(names, dict):
            label = names.get(
                class_id,
                f"class_{class_id}"
            )
        else:
            label = (
                names[class_id]
                if class_id < len(names)
                else f"class_{class_id}"
            )

        leaf_boxes.append({
            "bbox": bbox,
            "label": str(label),
            "confidence": confidence,
            "is_leaf": True
        })

    return leaf_boxes


# ==========================================================
# EMPTY RESULT
# ==========================================================

def empty_result():

    return {
        "disease": "Invalid Image",
        "species": "Unknown",
        "condition": "Invalid Image",
        "confidence": 0.0,

        "labeled_frame": None,
        "cropped_leaf": None,
        "all_objects_frame": None,
        "bounding_box_image": None,

        "detected_objects": [],

        "low_confidence": False,
        "plant_locked": False,
        "leaf_count": 0,

        "pest": None,
        "pest_confidence": 0.0,
        "pest_model_available": pest_model is not None,
        "pest_status": (
            "Classifier loaded; pest detector not configured"
            if pest_model is not None
            else "Pest classifier unavailable"
        ),

        "green_ratio": 0.0
    }


# ==========================================================
# MAIN DISEASE PREDICTION PIPELINE
# ==========================================================

def predict_disease_pipeline(image_path):

    result_empty = empty_result()

    frame = cv2.imread(image_path)

    if frame is None:
        return result_empty

    h, w = frame.shape[:2]

    # ------------------------------------------------------
    # STEP 1: GENERAL YOLO DETECTION
    # ------------------------------------------------------

    print("[AI] Running general object detection...", flush=True)

    general_boxes, plant_bbox = detect_general_objects(frame)

    # ------------------------------------------------------
    # STEP 2: LEAF DETECTION
    # ------------------------------------------------------

    print("[AI] Running dedicated leaf detection...", flush=True)

    leaf_boxes = detect_leaves(frame)

    # Prefer the highest-confidence dedicated leaf detection.
    best_leaf = max(
        leaf_boxes,
        key=lambda item: item["confidence"],
        default=None
    )

    if best_leaf is not None:
        plant_bbox = best_leaf["bbox"]

    plant_locked = plant_bbox is not None

    all_boxes = general_boxes + leaf_boxes

    detected_objects = [
        {
            "label": item["label"],
            "confidence": round(
                float(item["confidence"]) * 100,
                1
            ),
            "bbox": list(item["bbox"])
        }
        for item in all_boxes
    ]

    all_objects_frame = draw_all_detections(
        frame,
        all_boxes
    )

    # ------------------------------------------------------
    # STEP 3: SELECT PLANT / LEAF CROP
    # ------------------------------------------------------

    if not plant_locked:

        lock_box = (0, 0, w - 1, h - 1)

        labeled_frame = draw_tech_borders(
            frame,
            *lock_box,
            "NO PLANT OR LEAF DETECTED"
        )

        return {
            **result_empty,
            "disease": UNRECOGNIZED_LABEL,
            "condition": UNRECOGNIZED_LABEL,
            "labeled_frame": labeled_frame,
            "all_objects_frame": all_objects_frame,
            "bounding_box_image": all_objects_frame,
            "detected_objects": detected_objects,
            "low_confidence": True,
            "leaf_count": len(leaf_boxes)
        }

    x1, y1, x2, y2 = plant_bbox

    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.08)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)

    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)

    cropped_leaf = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if cropped_leaf.size == 0:
        return {
            **result_empty,
            "all_objects_frame": all_objects_frame,
            "bounding_box_image": all_objects_frame,
            "detected_objects": detected_objects,
            "leaf_count": len(leaf_boxes)
        }

    # ------------------------------------------------------
    # STEP 4: VEGETATION SANITY CHECK
    # ------------------------------------------------------

    hsv_crop = cv2.cvtColor(
        cropped_leaf,
        cv2.COLOR_BGR2HSV
    )

    green_mask = cv2.inRange(
        hsv_crop,
        (25, 30, 30),
        (95, 255, 255)
    )

    ch, cw = green_mask.shape

    y_start = int(ch * 0.2)
    y_end = max(int(ch * 0.8), 1)

    x_start = int(cw * 0.2)
    x_end = max(int(cw * 0.8), 1)

    center_mask = green_mask[
        y_start:y_end,
        x_start:x_end
    ]

    green_ratio = (
        float(np.count_nonzero(center_mask))
        / max(center_mask.size, 1)
    )

    if green_ratio < GREEN_RATIO_MIN:

        labeled_frame = draw_tech_borders(
            frame,
            *plant_bbox,
            f"LOW VEGETATION COLOR ({green_ratio * 100:.1f}%)"
        )

        return {
            **result_empty,
            "disease": UNRECOGNIZED_LABEL,
            "condition": UNRECOGNIZED_LABEL,
            "labeled_frame": labeled_frame,
            "cropped_leaf": cropped_leaf,
            "all_objects_frame": all_objects_frame,
            "bounding_box_image": all_objects_frame,
            "detected_objects": detected_objects,
            "low_confidence": True,
            "plant_locked": True,
            "leaf_count": len(leaf_boxes),
            "green_ratio": green_ratio
        }

    # ------------------------------------------------------
    # STEP 5: SAVE CROP FOR DISEASE CLASSIFICATION
    # ------------------------------------------------------

    if not cv2.imwrite(TEMP_CROP_PATH, cropped_leaf):

        raise RuntimeError(
            "Could not save the selected leaf crop."
        )

    try:

        processed_img = preprocess_image(
            TEMP_CROP_PATH
        )

        print("[AI] Running disease classifier...", flush=True)

        predictions = disease_model.predict(
            processed_img,
            verbose=0
        )

    finally:

        if os.path.exists(TEMP_CROP_PATH):
            try:
                os.remove(TEMP_CROP_PATH)
            except OSError:
                pass

    # ------------------------------------------------------
    # STEP 6: DISEASE CLASSIFICATION
    # ------------------------------------------------------

    probabilities = np.asarray(predictions[0])

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    if predicted_index < len(CLASS_NAMES):
        disease_name = str(
            CLASS_NAMES[predicted_index]
        )
    else:
        disease_name = f"class_{predicted_index}"

    species, condition = split_species_condition(
        disease_name
    )

    low_confidence = (
        confidence < DISEASE_MIN_CONFIDENCE
    )

    if low_confidence:

        disease_output = UNRECOGNIZED_LABEL
        condition_output = UNRECOGNIZED_LABEL

        display_label = (
            f"LOW DISEASE CONFIDENCE "
            f"({confidence * 100:.1f}%)"
        )

    else:

        disease_output = disease_name
        condition_output = condition

        display_label = f"{species}: {condition}"

    labeled_frame = draw_tech_borders(
        frame,
        *plant_bbox,
        display_label
    )

    # ------------------------------------------------------
    # STEP 7: FINAL RESULT
    # ------------------------------------------------------

    return {
        "disease": disease_output,
        "species": species,
        "condition": condition_output,
        "confidence": confidence,

        "labeled_frame": labeled_frame,
        "cropped_leaf": cropped_leaf,
        "all_objects_frame": all_objects_frame,
        "bounding_box_image": all_objects_frame,

        "detected_objects": detected_objects,

        "low_confidence": low_confidence,
        "plant_locked": plant_locked,
        "leaf_count": len(leaf_boxes),

        # IP102 classifier is available, but no pest detector
        # currently identifies insect crops.
        "pest": None,
        "pest_confidence": 0.0,
        "pest_model_available": pest_model is not None,
        "pest_status": (
            "Classifier loaded; pest bounding-box detector not configured"
            if pest_model is not None
            else "Pest classifier unavailable"
        ),

        "green_ratio": green_ratio
    }


# ==========================================================
# LIVE CAMERA FRAME PIPELINE
# ==========================================================

def predict_frame_pipeline(frame):

    if (
        frame is None
        or not hasattr(frame, "size")
        or frame.size == 0
    ):
        return empty_result()

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

        if not cv2.imwrite(temp_path, frame):

            raise RuntimeError(
                "Could not save the live camera frame."
            )

        return predict_disease_pipeline(temp_path)

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except OSError:
                pass