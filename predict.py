import os
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from utils.preprocessing import preprocess_image
from utils.class_names import CLASS_NAMES
import config

disease_model = tf.keras.models.load_model(config.DISEASE_MODEL_PATH)
yolo_model = YOLO(config.YOLO_GENERAL_MODEL_PATH)
COCO_NAMES = yolo_model.names

weed_model = YOLO(config.WEED_MODEL_PATH) if config.ENABLE_WEED_DETECTION else None
if weed_model is not None:
    print(f"[predict] Weed detection ENABLED — using {config.WEED_MODEL_PATH}")
else:
    print(f"[predict] Weed detection disabled — no model found at {config.WEED_MODEL_PATH}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_CROP_PATH = os.path.join(TEMP_DIR, "current_crop.jpg")

# ==========================================================
# NEW: confidence floor for the disease classifier.
# Add this same constant to config.py so it's tunable in one place
# like everything else:
#     DISEASE_MIN_CONFIDENCE = 0.55
# Below this threshold, the model's top guess is NOT trustworthy
# enough to report as a diagnosis — it gets flagged instead of
# silently reported (and silently written to the database) as if
# it were a confident answer.
# ==========================================================
DISEASE_MIN_CONFIDENCE = getattr(config, "DISEASE_MIN_CONFIDENCE", 0.55)
UNRECOGNIZED_LABEL = "Unrecognized / Low Confidence"


def split_species_condition(disease_name):
    name = disease_name
    for sep in ("___", "__", "-"):
        if sep in name:
            species_raw, condition_raw = name.split(sep, 1)
            species = species_raw.replace("_", " ").strip()
            condition = condition_raw.replace("_", " ").strip()
            return species, condition
    if "_" in name:
        species_raw, condition_raw = name.split("_", 1)
        return species_raw.replace("_", " ").strip(), condition_raw.replace("_", " ").strip()
    return "Unspecified Species", name.replace("_", " ")


def draw_tech_borders(img, x1, y1, x2, y2, label):
    color = (0, 255, 0)
    thickness = 2
    line_length = 20
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness + 2)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness + 2)
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness + 2)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness + 2)
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness + 2)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness + 2)
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness + 2)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness + 2)

    # ==========================================================
    # Readable label: green text directly on a green leaf was
    # illegible (text blended into the background). Draw a solid
    # dark background strip behind the text first, then white text
    # on top — readable regardless of what's behind it.
    # ==========================================================
    label_text = label.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

    text_x = x1
    text_y = max(20, y1 - 10)
    padding = 4
    cv2.rectangle(
        img,
        (text_x - padding, text_y - text_h - padding),
        (text_x + text_w + padding, text_y + baseline + padding),
        (0, 0, 0),
        -1,
    )
    cv2.putText(img, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return img


def draw_all_detections(img, boxes):
    out = img.copy()
    for det in boxes:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 100, 255) if det.get("is_weed") else (255, 200, 60)
        label = f'{det["label"]} {int(det["confidence"] * 100)}%'
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def detect_weeds(frame):
    if weed_model is None:
        return [], 0
    results = weed_model(frame, verbose=False)[0]
    weed_boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < config.WEED_MIN_CONFIDENCE:
            continue
        xyxy = tuple(box.xyxy[0].cpu().numpy().astype(int))
        weed_boxes.append({"bbox": xyxy, "label": "weed", "confidence": conf, "is_weed": True})
    return weed_boxes, len(weed_boxes)


def predict_disease_pipeline(image_path):
    empty_result = {
        "disease": "Invalid Image", "species": "Unknown", "condition": "Invalid Image",
        "confidence": 0.0, "labeled_frame": None, "cropped_leaf": None,
        "all_objects_frame": None, "detected_objects": [], "weed_count": 0,
        "low_confidence": False,
    }

    frame = cv2.imread(image_path)
    if frame is None:
        return empty_result

    h, w, _ = frame.shape
    yolo_results = yolo_model(frame, verbose=False)[0]

    all_boxes = []
    plant_bbox = None
    best_plant_conf = 0.0

    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < config.GENERAL_MIN_CONFIDENCE:
            continue
        xyxy = tuple(box.xyxy[0].cpu().numpy().astype(int))
        label = COCO_NAMES.get(cls, f"class_{cls}")
        all_boxes.append({"bbox": xyxy, "label": label, "confidence": conf, "is_weed": False})
        if cls == config.PLANT_CLASS_ID and conf > config.PLANT_MIN_CONFIDENCE and conf > best_plant_conf:
            plant_bbox = xyxy
            best_plant_conf = conf

    weed_boxes, weed_count = detect_weeds(frame)
    all_boxes.extend(weed_boxes)

    detected_objects = [
        {"label": b["label"], "confidence": round(b["confidence"] * 100, 1)} for b in all_boxes
    ]
    all_objects_frame = draw_all_detections(frame, all_boxes)

    # NOTE: this flag matters downstream. If YOLO never locks a proper
    # plant/leaf box, we're about to classify the WHOLE raw frame
    # (soil, rover chassis, background clutter and all) — which is a
    # major silent source of "wrong answers". We surface that instead
    # of hiding it.
    plant_locked = plant_bbox is not None

    if plant_locked:
        x1, y1, x2, y2 = plant_bbox
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        crop_x2, crop_y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        cropped_leaf = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        lock_box = (x1, y1, x2, y2)
    else:
        lock_box = (0, 0, w, h)
        cropped_leaf = frame

    if cropped_leaf is None or cropped_leaf.size == 0:
        cropped_leaf = frame

    ok = cv2.imwrite(TEMP_CROP_PATH, cropped_leaf)
    if not ok:
        return {**empty_result, "all_objects_frame": all_objects_frame,
                "detected_objects": detected_objects, "weed_count": weed_count}

    # ==========================================================
    # SANITY CHECK: does this even look like plant material?
    # The disease classifier is a closed-set softmax model — given ANY
    # image (a face, a wall, a shoe) it will still confidently pick
    # whichever of its 51 known classes looks "closest", even if the
    # image has nothing to do with a plant. Removing the plant_locked
    # requirement earlier fixed false rejections on real leaf photos,
    # but it also removed the only thing stopping totally unrelated
    # images from reaching the classifier at all.
    #
    # This is a cheap, model-free heuristic: real leaf/plant photos are
    # overwhelmingly green/vegetation-colored in HSV space. A human
    # face, wall, or random object is not. If green content is too low,
    # skip the classifier entirely and report no plant detected —
    # rather than let the classifier force a confident wrong guess.
    # Tune GREEN_RATIO_MIN in config.py if real leaf photos (unusual
    # lighting, mostly-brown diseased tissue, etc.) start getting
    # rejected by this check.
    # ==========================================================
    hsv_crop = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2HSV)
    # Broad "vegetation" hue range in OpenCV's HSV (H: 0-179): covers
    # yellow-green through blue-green, with generous saturation/value
    # floors so it still catches dull, dry, or diseased (brownish-green)
    # leaf tissue, not just vivid healthy green.
    green_mask = cv2.inRange(hsv_crop, (25, 30, 30), (95, 255, 255))

    # Measure green content in the CENTER region only, not the whole
    # crop. Whole-frame green% was too easy to fool — a person or
    # object photographed in front of foliage/greenery can score high
    # on background green alone, even though the actual subject in the
    # middle of the frame isn't a plant at all. Restricting the check
    # to the central ~60% of the frame (where the locked/cropped
    # subject should be) makes background greenery much less able to
    # rescue a non-plant subject.
    ch, cw = green_mask.shape
    y0, y1c = int(ch * 0.2), int(ch * 0.8)
    x0, x1c = int(cw * 0.2), int(cw * 0.8)
    center_mask = green_mask[y0:y1c, x0:x1c]
    green_ratio = float(np.count_nonzero(center_mask)) / max(center_mask.size, 1)
    GREEN_RATIO_MIN = getattr(config, "GREEN_RATIO_MIN", 0.20)

    if green_ratio < GREEN_RATIO_MIN:
        labeled_frame = draw_tech_borders(
            frame.copy(), *lock_box,
            f"NO PLANT DETECTED ({green_ratio * 100:.0f}% green)"
        )
        return {
            "disease": UNRECOGNIZED_LABEL,
            "species": "Unknown",
            "condition": UNRECOGNIZED_LABEL,
            "confidence": 0.0,
            "labeled_frame": labeled_frame,
            "cropped_leaf": cropped_leaf,
            "all_objects_frame": all_objects_frame,
            "detected_objects": detected_objects,
            "weed_count": weed_count,
            "low_confidence": True,
            "plant_locked": plant_locked,
        }

    processed_img = preprocess_image(TEMP_CROP_PATH)
    predictions = disease_model.predict(processed_img, verbose=False)
    pred_probabilities = predictions[0]
    predicted_index = int(np.argmax(pred_probabilities))

    disease_name = CLASS_NAMES[predicted_index]
    confidence_score = float(pred_probabilities[predicted_index])
    species, condition = split_species_condition(disease_name)

    # ==========================================================
    # Confidence-gate the final answer — based ONLY on the disease
    # classifier's own confidence, not on whether YOLO's generic COCO
    # "potted plant" class happened to fire.
    #
    # NOTE: plant_locked is intentionally NOT part of this condition
    # anymore. COCO's "potted plant" class is trained on full potted
    # houseplants, not close-up leaf macro shots — on real greenhouse/
    # rover photos it almost never fires, which was previously forcing
    # every single scan into "Unrecognized" regardless of how confident
    # the actual disease classifier was. plant_locked is still tracked
    # and returned (useful for debugging / UI display), it just no
    # longer gates the diagnosis itself.
    # ==========================================================
    is_uncertain = confidence_score < DISEASE_MIN_CONFIDENCE

    if is_uncertain:
        labeled_frame = draw_tech_borders(
            frame.copy(), *lock_box,
            f"LOW CONFIDENCE ({confidence_score * 100:.0f}%)"
        )
        return {
            "disease": UNRECOGNIZED_LABEL,
            "species": species,          # kept for reference/debugging in the UI
            "condition": UNRECOGNIZED_LABEL,
            "confidence": confidence_score,
            "labeled_frame": labeled_frame,
            "cropped_leaf": cropped_leaf,
            "all_objects_frame": all_objects_frame,
            "detected_objects": detected_objects,
            "weed_count": weed_count,
            "low_confidence": True,
            "plant_locked": plant_locked,
        }

    labeled_frame = draw_tech_borders(frame.copy(), *lock_box, f"{species}: {condition}")

    return {
        "disease": disease_name,
        "species": species,
        "condition": condition,
        "confidence": confidence_score,
        "labeled_frame": labeled_frame,
        "cropped_leaf": cropped_leaf,
        "all_objects_frame": all_objects_frame,
        "detected_objects": detected_objects,
        "weed_count": weed_count,
        "low_confidence": False,
        "plant_locked": plant_locked,
    }