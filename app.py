import base64
import html
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st
import config

from services.live_monitor import LiveMonitor
from services.live_inference import LiveInferenceWorker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgroVision Rover HUD",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

try:
    with open("style.css", "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )
except FileNotFoundError:
    pass


st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
      --av-bg: #07110d;
      --av-panel: #0d1b14;
      --av-panel-2: #102319;
      --av-line: rgba(148, 163, 184, .16);
      --av-green: #7ee787;
      --av-green-2: #28c76f;
      --av-muted: #91a69a;
      --av-text: #edf7ef;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }

    .stApp {
      background:
        radial-gradient(
          ellipse at 12% 0%,
          rgba(34,197,94,.12),
          transparent 34%
        ),
        radial-gradient(
          ellipse at 92% 12%,
          rgba(20,184,166,.08),
          transparent 30%
        ),
        linear-gradient(
          145deg,
          #07110d 0%,
          #091710 48%,
          #07110d 100%
        );
      color: var(--av-text);
    }

    [data-testid="stHeader"] {
      background: rgba(7,17,13,.72);
      backdrop-filter: blur(14px);
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(
        180deg,
        #0b1912 0%,
        #08130e 100%
      );
      border-right: 1px solid var(--av-line);
    }

    [data-testid="stSidebar"] > div:first-child {
      padding-top: 1.4rem;
    }

    .block-container {
      max-width: 1580px;
      padding-top: 2rem;
      padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
      font-family: 'Space Grotesk', sans-serif !important;
      letter-spacing: -.035em;
      color: #f2fbf4 !important;
    }

    h2 {
      font-size: 1.65rem !important;
    }

    h3 {
      font-size: 1.2rem !important;
    }

    p, label, [data-testid="stMarkdownContainer"] {
      color: #d6e5da;
    }

    .hero {
      position: relative;
      overflow: hidden;
      padding: 30px 34px;
      margin: 0 0 24px 0;
      border: 1px solid rgba(126,231,135,.24);
      border-radius: 22px;
      background:
        linear-gradient(
          110deg,
          rgba(16,42,27,.97),
          rgba(9,27,19,.92)
        ),
        radial-gradient(
          circle at 85% 20%,
          rgba(126,231,135,.16),
          transparent 35%
        );
      box-shadow:
        0 18px 55px rgba(0,0,0,.22),
        inset 0 1px 0 rgba(255,255,255,.04);
    }

    .hero:after {
      content: '';
      position: absolute;
      width: 260px;
      height: 260px;
      right: -70px;
      top: -120px;
      border: 1px solid rgba(126,231,135,.17);
      border-radius: 50%;
      box-shadow:
        0 0 0 28px rgba(126,231,135,.025),
        0 0 0 58px rgba(126,231,135,.02);
    }

    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 11px;
      border: 1px solid rgba(126,231,135,.28);
      border-radius: 99px;
      background: rgba(34,197,94,.10);
      color: #9af5a5;
      font: 700 10px 'DM Sans', sans-serif;
      letter-spacing: .16em;
    }

    .hero-badge:before {
      content: '';
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #7ee787;
      box-shadow: 0 0 12px #7ee787;
    }

    .hero .title {
      position: relative;
      z-index: 1;
      margin-top: 13px;
      color: #f4fff6;
      font: 700 clamp(2rem, 4vw, 3.15rem)
        'Space Grotesk', sans-serif;
      letter-spacing: -.055em;
    }

    .hero .subtitle {
      position: relative;
      z-index: 1;
      margin-top: 5px;
      color: #a8c1ae;
      font-size: .98rem;
    }

    .sidebar-title {
      font: 700 1.2rem 'Space Grotesk', sans-serif;
      color: #eaffed;
    }

    .sidebar-label {
      color: #8fa99a;
      font-size: .68rem;
      font-weight: 700;
      letter-spacing: .13em;
    }

    .sidebar-info {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--av-line);
      border-radius: 12px;
      background: rgba(255,255,255,.025);
      color: #a8c1ae;
      font-size: .72rem;
      overflow-wrap: anywhere;
      line-height: 1.8;
    }

    div[data-testid="stMetric"] {
      padding: 15px 17px;
      border: 1px solid var(--av-line);
      border-radius: 15px;
      background: linear-gradient(
        145deg,
        rgba(17,39,26,.94),
        rgba(10,25,17,.94)
      );
      box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }

    div[data-testid="stMetric"] label {
      color: #91aa99 !important;
      font-size: .76rem !important;
      font-weight: 600 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: #eaffed !important;
      font-family: 'Space Grotesk', sans-serif;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--av-line) !important;
      border-radius: 17px !important;
      background: rgba(13,27,20,.62);
    }

    [data-testid="stImage"] img {
      border-radius: 15px;
      border: 1px solid rgba(126,231,135,.16);
      box-shadow: 0 12px 32px rgba(0,0,0,.2);
    }

    .stButton > button {
      min-height: 2.75rem;
      border: 1px solid rgba(126,231,135,.28);
      border-radius: 11px;
      color: #eaffed;
      background: linear-gradient(
        135deg,
        #176b3a,
        #124d30
      );
      font-weight: 700;
      transition: all .18s ease;
      box-shadow: 0 7px 20px rgba(10,90,44,.15);
    }

    .stButton > button:hover {
      border-color: #7ee787;
      color: #fff;
      background: linear-gradient(
        135deg,
        #21894b,
        #17643a
      );
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(34,197,94,.18);
    }

    .stButton > button:disabled {
      background: #17251c;
      color: #718579;
      border-color: #26372c;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      background: #0b1912;
      border-color: #294333;
      border-radius: 10px;
      color: #eaffed;
    }

    [data-testid="stFileUploader"] {
      padding: 12px;
      border: 1px dashed rgba(126,231,135,.35);
      border-radius: 15px;
      background: rgba(16,42,27,.34);
    }

    [data-testid="stExpander"] {
      border: 1px solid var(--av-line);
      border-radius: 13px;
      background: rgba(13,27,20,.5);
    }

    [data-testid="stAlert"] {
      border-radius: 12px;
    }

    hr {
      border-color: var(--av-line) !important;
    }

    .footer {
      margin-top: 42px;
      padding: 16px 0 4px;
      border-top: 1px solid var(--av-line);
      color: #71877a;
      text-align: center;
      font-size: .75rem;
      letter-spacing: .04em;
    }

    @media (max-width: 800px) {
      .hero {
        padding: 23px 20px;
      }

      .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">ROVER OPS · LIVE</div>

      <div class="title">
        AgroVision Rover HUD
      </div>

      <div class="subtitle">
        INTELLIGENT CROP MONITORING
        • LIVE VISION
        • AI-POWERED DIAGNOSTICS
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_BASE = "http://127.0.0.1:8001"

PREDICT_URL = f"{API_BASE}/predict"
ASK_URL = f"{API_BASE}/ask"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "esp32_ip": "192.168.1.37",
    "gps_ip": "192.168.1.50",
    "live_ai_enabled": False,
    "live_monitor": None,
    "live_inference_worker": None,
    "frozen_frame": None,
    "api_results": None,
    "manual_api_results": None,
    "last_mode": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# IMAGE HELPERS
# ============================================================

def is_image_array(value):
    return (
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim in (2, 3)
    )


def decode_data_url(value):
    """
    Decode a data:image/...;base64,... string
    into an OpenCV BGR image.
    """

    if (
        not isinstance(value, str)
        or not value.startswith("data:image")
    ):
        return None

    try:
        encoded = value.split(",", 1)[1]
        raw = base64.b64decode(encoded)

        return cv2.imdecode(
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None


def to_bgr(value):
    """
    Accept either an OpenCV ndarray
    or an API base64 data URL.
    """

    if is_image_array(value):

        image = value

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        return image.copy()

    if isinstance(value, str):
        return decode_data_url(value)

    return None


# ============================================================
# DIGITAL ZOOM
# ============================================================

def zoom_crop(
    image,
    zoom_factor=1.0,
    detected_objects=None,
    auto_zoom=False,
):
    """
    Software display zoom.

    If auto zoom is enabled, it attempts to crop around
    the largest detected plant/leaf bounding box.

    If no bounding box is available, it uses a center crop.

    This does not physically move the camera lens.
    """

    if not is_image_array(image):
        return image

    h, w = image.shape[:2]

    factor = float(zoom_factor)

    target_box = None

    if auto_zoom:

        candidates = []

        for obj in (detected_objects or []):

            if not isinstance(obj, dict):
                continue

            label = str(
                obj.get("label", "")
            ).lower()

            bbox = obj.get("bbox")

            if bbox is None or len(bbox) != 4:
                continue

            if "leaf" in label or "plant" in label:

                try:

                    x1, y1, x2, y2 = [
                        int(v) for v in bbox
                    ]

                    area = (
                        max(0, x2 - x1)
                        * max(0, y2 - y1)
                    )

                    if area > 0:
                        candidates.append(
                            (
                                area,
                                (x1, y1, x2, y2),
                            )
                        )

                except (TypeError, ValueError):
                    continue

        if candidates:

            target_box = max(
                candidates,
                key=lambda item: item[0],
            )[1]

            x1, y1, x2, y2 = target_box

            box_w = x2 - x1
            box_h = y2 - y1

            crop_w = min(
                w,
                max(box_w * 1.7, w / 2.0),
            ) / max(1.0, factor)

            crop_h = min(
                h,
                max(box_h * 1.7, h / 2.0),
            ) / max(1.0, factor)

            crop_w = max(
                1,
                min(w, int(crop_w)),
            )

            crop_h = max(
                1,
                min(h, int(crop_h)),
            )

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            x1 = int(
                max(
                    0,
                    min(w - crop_w, cx - crop_w / 2),
                )
            )

            y1 = int(
                max(
                    0,
                    min(h - crop_h, cy - crop_h / 2),
                )
            )

            x2 = int(x1 + crop_w)
            y2 = int(y1 + crop_h)

        else:

            crop_w = max(
                1,
                int(w / max(1.0, factor)),
            )

            crop_h = max(
                1,
                int(h / max(1.0, factor)),
            )

            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2

            x2 = x1 + crop_w
            y2 = y1 + crop_h

    elif factor <= 1.001:

        return image

    else:

        crop_w = max(
            1,
            int(w / factor),
        )

        crop_h = max(
            1,
            int(h / factor),
        )

        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2

        x2 = x1 + crop_w
        y2 = y1 + crop_h

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return image

    return cv2.resize(
        crop,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# IMAGE DISPLAY
# ============================================================

def show_cv_image(
    image,
    caption=None,
    key=None,
    use_container_width=True,
):

    if not is_image_array(image):

        st.warning(
            "No valid image is available to display."
        )

        return

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    st.image(
        rgb,
        caption=caption,
        channels="RGB",
        use_container_width=use_container_width,
        output_format="PNG",
    )


# ============================================================
# CONFIDENCE CONVERSION
# ============================================================

def confidence_percent(value):

    try:

        number = float(value or 0)

        if 0 <= number <= 1:
            return round(number * 100, 2)

        return round(number, 2)

    except (TypeError, ValueError):

        return 0.0


# ============================================================
# GPS
# ============================================================

def get_current_gps():

    if not getattr(
        config,
        "ENABLE_GPS_TRACKING",
        False,
    ):

        return None, None

    try:

        response = requests.get(
            f"http://{st.session_state.gps_ip.strip()}/gps",
            timeout=3,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("valid"):

            return (
                data.get("lat"),
                data.get("lng"),
            )

    except requests.RequestException:
        pass

    return None, None


# ============================================================
# DETECTION RESULTS
# ============================================================

def render_detection_list(
    result,
    prefix,
):

    objects = result.get(
        "detected_objects"
    ) or []

    if objects:

        with st.expander(
            f"Detected objects ({len(objects)})",
            expanded=True,
        ):

            for index, obj in enumerate(objects):

                label = obj.get(
                    "label",
                    "Unknown",
                )

                score = confidence_percent(
                    obj.get("confidence", 0)
                )

                st.write(
                    f"• {label} — {score:.1f}%"
                )

    else:

        st.info(
            "No objects passed the detection thresholds."
        )

    pest = result.get("pest")

    if pest:

        st.write(
            f"**Pest classification:** {pest} "
            f"({confidence_percent(result.get('pest_confidence')):.1f}%)"
        )

    pest_status = result.get("pest_status")

    if pest_status:
        st.caption(pest_status)


# ============================================================
# AI ADVISOR
# ============================================================

def render_ask_box(
    species,
    condition,
    key_prefix,
):

    st.markdown(
        "### Ask AgroVision AI Advisor"
    )

    question = st.text_input(
        "Ask a question about this diagnosis",
        key=f"{key_prefix}_question",
        placeholder=(
            "Ask about symptoms, prevention or treatment..."
        ),
    )

    if st.button(
        "Ask AI Advisor",
        key=f"{key_prefix}_ask",
    ):

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            with st.spinner(
                "Contacting AgroVision AI Advisor..."
            ):

                try:

                    response = requests.post(
                        ASK_URL,
                        json={
                            "species": str(species),
                            "condition": str(condition),
                            "question": question.strip(),
                        },
                        timeout=90,
                    )

                    response.raise_for_status()

                    st.session_state[
                        f"{key_prefix}_answer"
                    ] = response.json().get(
                        "answer",
                        "The advisor returned no answer.",
                    )

                except requests.RequestException as exc:

                    st.error(
                        f"Could not contact the AI advisor: {exc}"
                    )

    answer = st.session_state.get(
        f"{key_prefix}_answer"
    )

    if answer:
        st.info(answer)


# ============================================================
# DIAGNOSIS PANEL
# ============================================================

def render_diagnosis_panel(
    result,
    prefix="diagnosis",
):

    if not result:

        st.info(
            "No diagnosis available yet."
        )

        return

    species = result.get(
        "species",
        "Unknown",
    )

    condition = (
        result.get("condition")
        or result.get("disease", "Unknown")
    )

    confidence = confidence_percent(
        result.get("confidence", 0)
    )

    severity = result.get(
        "severity",
        "N/A",
    )

    recommendations = (
        result.get("recommendations")
        or []
    )

    treatment = result.get(
        "treatment",
        "No recommended action returned.",
    )

    st.markdown(
        "### 🔍 Diagnostics Evaluation"
    )

    if (
        confidence > 0
        and condition not in (
            "Unrecognized / Low Confidence",
            "Invalid Image",
            "No Target Object Spotted",
        )
    ):

        st.success(
            "TARGET DETECTED"
        )

    else:

        st.warning(
            "NO CONFIDENT TARGET DIAGNOSIS"
        )

    c1, c2 = st.columns(2)

    c1.metric(
        "Species",
        species,
    )

    c2.metric(
        "Condition",
        condition,
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Confidence",
        f"{confidence:.2f}%",
    )

    c4.metric(
        "Severity",
        severity,
    )

    render_detection_list(
        result,
        prefix,
    )

    if result.get("leaf_count") is not None:

        st.metric(
            "Detected leaves",
            result.get("leaf_count", 0),
        )

    st.markdown(
        "**Recommendations**"
    )

    if recommendations:

        for item in recommendations:
            st.write(f"• {item}")

    else:

        st.write(
            "No specific recommendations available."
        )

    st.markdown(
        "**Recommended action**"
    )

    st.write(treatment)

    render_ask_box(
        species,
        condition,
        prefix,
    )


# ============================================================
# LIVE AI DISPLAY
# ============================================================

@st.fragment(run_every=1.0)
def render_live_ai_feed():

    worker = st.session_state.get(
        "live_inference_worker"
    )

    monitor = st.session_state.get(
        "live_monitor"
    )

    if not st.session_state.get(
        "live_ai_enabled"
    ):

        st.info(
            "Live AI is stopped."
        )

        return

    if worker is None or monitor is None:

        st.warning(
            "Live AI services are not initialized."
        )

        return

    camera_status = (
        monitor.get_status() or {}
    )

    worker_status = (
        worker.get_status() or {}
    )

    if camera_status.get("error"):

        st.error(
            f"Camera error: {camera_status['error']}"
        )

    if worker_status.get("error"):

        st.error(
            f"Inference error: {worker_status['error']}"
        )

    st.caption(
        "Camera frames received: "
        f"{camera_status.get('frames_received', 0)}"
    )

    result = worker.get_latest_result()

    if result is None:

        st.info(
            "Waiting for the first AI inference result..."
        )

        return

    bbox_image = result.get(
        "bounding_box_image"
    )

    all_objects_image = result.get(
        "all_objects_frame"
    )

    labeled_image = result.get(
        "labeled_frame"
    )

    display_image = None

    for candidate in (
        bbox_image,
        all_objects_image,
        labeled_image,
    ):

        if is_image_array(candidate):

            display_image = candidate
            break

    detected_objects = (
        result.get("detected_objects")
        or []
    )

    st.markdown(
        "### 🧠 Live AI Analysis"
    )

    auto_zoom = st.checkbox(
        "Automatic plant/leaf zoom",
        value=True,
        key="live_auto_zoom",
        help=(
            "Zooms the displayed image around the largest "
            "detected plant/leaf box when bbox coordinates "
            "are available."
        ),
    )

    manual_zoom = st.slider(
        "Display zoom",
        min_value=1.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="live_zoom_level",
        help=(
            "Manual digital zoom of the displayed frame. "
            "1.0x is the original view."
        ),
    )

    if display_image is not None:

        if auto_zoom:

            zoomed = zoom_crop(
                display_image,
                zoom_factor=manual_zoom,
                detected_objects=detected_objects,
                auto_zoom=True,
            )

        else:

            zoomed = zoom_crop(
                display_image,
                zoom_factor=manual_zoom,
            )

        show_cv_image(
            zoomed,
            caption="Live AI annotated frame",
        )

    else:

        st.warning(
            "AI returned no annotated image for this frame."
        )

    render_detection_list(
        result,
        "live",
    )

    st.write(
        f"**Plant locked:** {result.get('plant_locked', False)}"
    )

    st.write(
        f"**Disease result:** {result.get('disease', 'Pending')}"
    )

    st.write(
        f"**Species:** {result.get('species', 'Unknown')}"
    )

    st.write(
        "**Disease confidence:** "
        f"{confidence_percent(result.get('confidence', 0)):.2f}%"
    )

    if result.get("green_ratio") is not None:

        st.write(
            "**Vegetation ratio:** "
            f"{float(result['green_ratio']):.1%}"
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    "<h2 class='sidebar-title'>🛰️ Navigation Panel</h2>",
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Select Rover Input Mode",
    [
        "ESP32 Live Camera",
        "Upload Image Asset",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<hr class='sidebar-divider'>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "### ESP32-CAM IP ADDRESS"
)

st.session_state.esp32_ip = st.sidebar.text_input(
    "Camera IP",
    value=st.session_state.esp32_ip,
    key="esp32_ip_input",
    help=(
        "Enter the IP printed in Arduino Serial Monitor. "
        "Example: 192.168.1.37"
    ),
)

camera_ip = (
    st.session_state.esp32_ip.strip()
    .replace("http://", "")
    .replace("https://", "")
    .rstrip("/")
)

ESP32_BASE = f"http://{camera_ip}"

STREAM_URL = f"{ESP32_BASE}:81/stream"

CAPTURE_URL = f"{ESP32_BASE}/capture"

st.sidebar.markdown(
    f"""
    <div class="sidebar-info">
      <div>
        <b>Stream:</b>
        {html.escape(STREAM_URL)}
      </div>

      <div>
        <b>Snapshot:</b>
        {html.escape(CAPTURE_URL)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if getattr(
    config,
    "ENABLE_GPS_TRACKING",
    False,
):

    st.sidebar.markdown(
        "### GPS MODULE IP ADDRESS"
    )

    st.session_state.gps_ip = st.sidebar.text_input(
        "GPS IP",
        value=st.session_state.gps_ip,
        key="gps_ip_input",
    )


# ============================================================
# MODE CHANGE RESET
# ============================================================

if st.session_state.last_mode != mode:

    st.session_state.frozen_frame = None

    st.session_state.api_results = None

    st.session_state.manual_api_results = None

    st.session_state.last_mode = mode


# ============================================================
# ESP32 LIVE CAMERA MODE
# ============================================================

if mode == "ESP32 Live Camera":

    st.markdown(
        "## 🛰️ Rover Main Track View"
    )

    # --------------------------------------------------------
    # CAMERA DISPLAY
    # --------------------------------------------------------

    if st.session_state.frozen_frame is None:

        st.markdown(
            f"""
            <div class="camera-box">
              <img
                src="{html.escape(STREAM_URL)}"
                style="width:100%; border-radius:12px;"
                onerror="this.style.opacity=0.3;"
              />
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        frozen = st.session_state.frozen_frame

        zoom = st.slider(
            "Snapshot display zoom",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="snapshot_zoom",
        )

        show_cv_image(
            zoom_crop(frozen, zoom),
            caption="Locked target snapshot",
        )

    # --------------------------------------------------------
    # CONTROL BUTTONS
    # --------------------------------------------------------

    start_col, stop_col, snap_col = st.columns(3)

    with start_col:

        start_clicked = st.button(
            "▶ Start Live AI",
            use_container_width=True,
            disabled=st.session_state.live_ai_enabled,
        )

    with stop_col:

        stop_clicked = st.button(
            "⏹ Stop Live AI",
            use_container_width=True,
            disabled=not st.session_state.live_ai_enabled,
        )

    with snap_col:

        snapshot_clicked = st.button(
            "📸 Lock Target & Scan",
            use_container_width=True,
            disabled=(
                st.session_state.frozen_frame is not None
            ),
        )

    # --------------------------------------------------------
    # START LIVE AI
    # --------------------------------------------------------

    if start_clicked:

        try:

            monitor = LiveMonitor(
                camera_ip=camera_ip
            )

            worker = LiveInferenceWorker(
                monitor
            )

            st.session_state.live_monitor = monitor

            st.session_state.live_inference_worker = worker

            monitor.start()

            worker.start()

            st.session_state.live_ai_enabled = True

            st.success(
                "Live AI started."
            )

            st.rerun()

        except Exception as exc:

            st.session_state.live_ai_enabled = False

            st.session_state.live_monitor = None

            st.session_state.live_inference_worker = None

            st.error(
                f"Could not start Live AI: {exc}"
            )

    # --------------------------------------------------------
    # STOP LIVE AI
    # --------------------------------------------------------

    if stop_clicked:

        try:

            worker = st.session_state.live_inference_worker

            monitor = st.session_state.live_monitor

            if worker is not None:
                worker.stop()

            if monitor is not None:
                monitor.stop()

        except Exception as exc:

            st.warning(
                f"Shutdown warning: {exc}"
            )

        st.session_state.live_inference_worker = None

        st.session_state.live_monitor = None

        st.session_state.live_ai_enabled = False

        st.success(
            "Live AI stopped."
        )

        st.rerun()

    # --------------------------------------------------------
    # LIVE AI FEED
    # --------------------------------------------------------

    if st.session_state.live_ai_enabled:

        st.divider()

        render_live_ai_feed()

    # --------------------------------------------------------
    # SNAPSHOT AND ANALYSIS
    # --------------------------------------------------------

    if snapshot_clicked:

        with st.spinner(
            "Capturing a snapshot from ESP32-CAM..."
        ):

            try:

                snap_response = requests.get(
                    CAPTURE_URL,
                    timeout=15,
                )

                snap_response.raise_for_status()

                image_bytes = np.frombuffer(
                    snap_response.content,
                    dtype=np.uint8,
                )

                frame = cv2.imdecode(
                    image_bytes,
                    cv2.IMREAD_COLOR,
                )

                if frame is None:

                    st.error(
                        "ESP32 returned an invalid image."
                    )

                else:

                    lat, lng = get_current_gps()

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                    )

                    if not ok:

                        st.error(
                            "Could not encode the camera snapshot."
                        )

                    else:

                        form_data = {}

                        if (
                            lat is not None
                            and lng is not None
                        ):

                            form_data = {
                                "latitude": str(lat),
                                "longitude": str(lng),
                            }

                        files = {
                            "file": (
                                "rover_snapshot.jpg",
                                encoded.tobytes(),
                                "image/jpeg",
                            )
                        }

                        with st.spinner(
                            "Running plant and disease diagnostics..."
                        ):

                            response = requests.post(
                                PREDICT_URL,
                                files=files,
                                data=form_data,
                                timeout=120,
                            )

                            response.raise_for_status()

                            st.session_state.api_results = (
                                response.json()
                            )

                            st.session_state.frozen_frame = frame

                            st.rerun()

            except requests.RequestException as exc:

                st.error(
                    f"Camera/API request failed: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"Snapshot processing failed: {exc}"
                )

    # --------------------------------------------------------
    # SNAPSHOT RESULTS
    # --------------------------------------------------------

    if st.session_state.frozen_frame is not None:

        if st.button(
            "🔄 Resume Live Camera Stream",
            use_container_width=True,
        ):

            st.session_state.frozen_frame = None

            st.session_state.api_results = None

            st.rerun()

        if st.session_state.api_results:

            result = st.session_state.api_results

            st.divider()

            left, right = st.columns([5, 4])

            with left:

                st.markdown(
                    "### 🎯 AI Detection Results"
                )

                annotated = (
                    result.get("bounding_box_image")
                    or result.get("all_objects_frame")
                    or result.get("labeled_frame")
                )

                decoded = (
                    decode_data_url(annotated)
                    if isinstance(annotated, str)
                    else None
                )

                if decoded is not None:

                    zoom = st.slider(
                        "Annotated image zoom",
                        1.0,
                        3.0,
                        1.0,
                        0.1,
                        key="snapshot_annotated_zoom",
                    )

                    show_cv_image(
                        zoom_crop(decoded, zoom)
                    )

                else:

                    st.info(
                        "The API did not return an annotated image."
                    )

            with right:

                render_diagnosis_panel(
                    result,
                    "snapshot",
                )


# ============================================================
# MANUAL IMAGE UPLOAD MODE
# ============================================================

elif mode == "Upload Image Asset":

    st.markdown(
        "## 📤 Manual Target Image Upload"
    )

    uploaded_file = st.file_uploader(
        "Upload a plant/leaf/pest image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="manual_upload",
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        frame = cv2.imdecode(
            np.frombuffer(
                file_bytes,
                np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            st.error(
                "This file is not a valid JPG/PNG image."
            )

        else:

            left, right = st.columns([5, 4])

            with left:

                st.markdown(
                    "### 🎯 Uploaded Target Asset"
                )

                show_cv_image(
                    frame,
                    caption="Original uploaded image",
                )

                if st.button(
                    "Run Manual Asset Diagnostics",
                    use_container_width=True,
                    key="run_manual_diagnostics",
                ):

                    with st.spinner(
                        "Analyzing uploaded image..."
                    ):

                        try:

                            response = requests.post(
                                PREDICT_URL,
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        file_bytes,
                                        uploaded_file.type
                                        or "image/jpeg",
                                    )
                                },
                                timeout=120,
                            )

                            response.raise_for_status()

                            st.session_state.manual_api_results = (
                                response.json()
                            )

                            st.rerun()

                        except requests.RequestException as exc:

                            st.error(
                                f"Could not connect to analysis API: {exc}"
                            )

            with right:

                result = st.session_state.manual_api_results

                if result:

                    st.markdown(
                        "### 🧠 AI Detection — Bounding Boxes"
                    )

                    annotated_value = (
                        result.get("bounding_box_image")
                        or result.get("all_objects_frame")
                        or result.get("labeled_frame")
                    )

                    annotated = (
                        decode_data_url(annotated_value)
                        if isinstance(annotated_value, str)
                        else None
                    )

                    if annotated is not None:

                        zoom = st.slider(
                            "Annotated image zoom",
                            min_value=1.0,
                            max_value=3.0,
                            value=1.0,
                            step=0.1,
                            key="manual_annotated_zoom",
                        )

                        show_cv_image(
                            zoom_crop(annotated, zoom)
                        )

                    render_diagnosis_panel(
                        result,
                        "manual",
                    )

                else:

                    st.info(
                        "Upload an image, then click "
                        "Run Manual Asset Diagnostics."
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class='footer'>
      AgroVision AI Operations HUD v3.0
      • Autonomous Farm Rover
    </div>
    """,
    unsafe_allow_html=True,
)