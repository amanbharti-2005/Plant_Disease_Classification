import streamlit as st
import cv2
import numpy as np
import requests
import base64
import config

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AgroVision Rover HUD", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("style.css file not found! Please ensure it is in the same directory.")

st.markdown("""
<div class='hero'>
    <div class='hero-badge'>ROVER OPS · LIVE</div>
    <div class='title'>AgroVision Rover HUD</div>
    <div class='subtitle'>Autonomous Field Object Tracking &amp; Plant Disease Diagnostics</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION & STATE INITIALIZATION
# ==========================================
FASTAPI_URL = "http://127.0.0.1:8001/predict"
ASK_URL = "http://127.0.0.1:8001/ask"

if "esp32_ip" not in st.session_state:
    st.session_state.esp32_ip = "192.168.1.38"
if "gps_ip" not in st.session_state:
    st.session_state.gps_ip = "192.168.1.50"
if "frozen_frame" not in st.session_state:
    st.session_state.frozen_frame = None
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "manual_api_results" not in st.session_state:
    st.session_state.manual_api_results = None


def get_image_base64(img_array_rgb):
    bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def get_current_gps():
    """Fetches the latest fix from the GPS ESP32's /gps endpoint.
    Returns (lat, lng) or (None, None) if disabled, unreachable, or no
    fix yet — scans still work without GPS, they just won't be geotagged."""
    if not config.ENABLE_GPS_TRACKING:
        return None, None

    gps_url = f"http://{st.session_state.gps_ip}/gps"
    try:
        resp = requests.get(gps_url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("valid"):
                return data.get("lat"), data.get("lng")
    except Exception:
        pass
    return None, None


# ==========================================================
# COMBINED SCAN IMAGE — one large image with the disease label
# overlay burned in by the backend (labeled_frame), plus a CSS-only
# scanning-line animation on top. No JS, no iframe — just a <style>
# keyframe animation, so this can't silently fail to render the way
# the old JS-driven component could.
# ==========================================================
def render_combined_scan_image(res, fallback_display_b64):
    display_img = res.get("labeled_frame") or res.get("all_objects_frame") or fallback_display_b64
    if not display_img:
        return

    st.markdown(f"""
    <style>
    @keyframes scanLineMove {{
        0%   {{ top: 0%; opacity: 1; }}
        90%  {{ top: 100%; opacity: 1; }}
        100% {{ top: 100%; opacity: 0; }}
    }}
    @keyframes lockedFadeIn {{
        0%, 80% {{ opacity: 0; transform: translateY(-6px); }}
        100%    {{ opacity: 1; transform: translateY(0); }}
    }}
    .scan-wrap {{
        position: relative;
        width: 100%;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #22C55E;
        box-shadow: 0 0 24px rgba(34,197,94,0.15);
    }}
    .scan-wrap img {{
        width: 100%;
        display: block;
    }}
    .scan-line {{
        position: absolute;
        left: 0;
        width: 100%;
        height: 3px;
        background: #00FF88;
        box-shadow: 0 0 16px 3px #00FF88;
        animation: scanLineMove 1.8s ease-in-out 1 forwards;
    }}
    .locked-badge {{
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #4ADE80;
        color: #4ADE80;
        padding: 5px 12px;
        border-radius: 100px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
        opacity: 0;
        animation: lockedFadeIn 2.2s ease forwards;
    }}
    </style>
    <div class="scan-wrap">
        <img src="{display_img}" />
        <div class="scan-line"></div>
        <div class="locked-badge">✓ TARGET LOCKED</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# DIAGNOSIS RESULT PANEL — pure native Streamlit widgets.
# No HTML/JS component, no iframe — this cannot silently fail to
# render the way the old components.html() cinematic version could.
# Slightly less flashy, but guaranteed to show every field every time.
# ==========================================================
def render_diagnosis_panel(res, fallback_display_b64, show_image=True):
    if not res:
        st.info("No scan results yet.")
        return

    species = res.get("species", "Unknown")
    condition = res.get("condition", res.get("disease", "Unknown"))
    confidence = res.get("confidence", 0) or 0
    severity = res.get("severity", "N/A")
    treatment = res.get("treatment", "No issues found.")
    recommendations = res.get("recommendations", []) or []
    detected_objects = res.get("detected_objects", []) or []

    found = confidence and confidence > 0
    status_label = "🟢 TARGET DETECTED" if found else "🟡 NO OBJECT LOCKED"
    st.markdown(f"**{status_label}**")

    if show_image:
        display_img = res.get("labeled_frame") or res.get("all_objects_frame") or fallback_display_b64
        if display_img:
            st.image(display_img, use_container_width=True)

    # Detected objects ticker
    if detected_objects:
        with st.expander(f"Objects detected ({len(detected_objects)})", expanded=False):
            for obj in detected_objects:
                st.write(f"▸ {obj.get('label', 'unknown')} — {obj.get('confidence', 0)}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Species", species)
        st.metric("Confidence", f"{confidence}%")
    with c2:
        st.metric("Condition", condition)
        st.metric("Severity", severity)

    st.markdown("**Recommendations**")
    if recommendations:
        for r in recommendations:
            st.write(f"• {r}")
    else:
        st.write("No specific recommendations.")

    st.markdown("**Recommended Action**")
    st.write(treatment)


def render_ask_box(species, condition, key_prefix):
    """A simple follow-up Q&A box under a diagnosis — calls /ask."""
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    question = st.text_input(
        "Ask a follow-up question about this diagnosis",
        key=f"{key_prefix}_question",
        placeholder="e.g. Where can I buy copper fungicide near me?"
    )
    if st.button("Ask", key=f"{key_prefix}_ask_btn"):
        if question.strip():
            with st.spinner("Consulting AI advisor..."):
                try:
                    resp = requests.post(
                        ASK_URL,
                        json={"species": species, "condition": condition, "question": question},
                        timeout=90
                    )
                    if resp.status_code == 200:
                        st.session_state[f"{key_prefix}_answer"] = resp.json().get("answer", "No answer returned.")
                    else:
                        st.error(f"API returned status {resp.status_code}")
                except Exception as e:
                    st.error(f"Could not reach AI advisor: {e}")

    if st.session_state.get(f"{key_prefix}_answer"):
        st.markdown(f"""
        <div class='card' style="margin-top:10px;">
            <div style="font-size:11px; letter-spacing:1px; color:#64748B; margin-bottom:6px;">AI ADVISOR</div>
            <div style="font-size:14px; color:#E2E8F0; line-height:1.6;">{st.session_state[f'{key_prefix}_answer']}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("<h2 class='sidebar-title'>🛰️ Navigation Panel</h2>", unsafe_allow_html=True)
mode = st.sidebar.radio("Select Rover Input Mode", ["ESP32 Live Camera", "Upload Image Asset"], label_visibility="collapsed")

st.sidebar.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-label'>ESP32-CAM IP ADDRESS</div>", unsafe_allow_html=True)
st.session_state.esp32_ip = st.sidebar.text_input(
    "ESP32 IP",
    value=st.session_state.esp32_ip,
    label_visibility="collapsed",
    help="Just the IP printed in the Arduino Serial Monitor, e.g. 10.201.79.249 (no http://, no port)."
)

ESP32_BASE = f"http://{st.session_state.esp32_ip}"
STREAM_URL = f"{ESP32_BASE}:81/stream"
CAPTURE_URL = f"{ESP32_BASE}/capture"

st.sidebar.markdown(f"""
<div class="sidebar-info">
    <div><b>Stream:</b> {STREAM_URL}</div>
    <div><b>Snapshot:</b> {CAPTURE_URL}</div>
</div>
""", unsafe_allow_html=True)

if config.ENABLE_GPS_TRACKING:
    st.sidebar.markdown("<div class='sidebar-label' style='margin-top:14px;'>GPS MODULE IP ADDRESS</div>", unsafe_allow_html=True)
    st.session_state.gps_ip = st.sidebar.text_input(
        "GPS IP",
        value=st.session_state.gps_ip,
        label_visibility="collapsed",
        help="IP printed by the GPS ESP32's Serial Monitor (the one running gps_server.ino)."
    )

if "last_mode" not in st.session_state or st.session_state.last_mode != mode:
    st.session_state.frozen_frame = None
    st.session_state.api_results = None
    st.session_state.last_mode = mode

# ==========================================
# INTERFACE EXECUTION LOGIC
# ==========================================

if mode == "ESP32 Live Camera":
    col1, col2 = st.columns([5, 4])

    with col1:
        st.markdown("<h3>🛰️ Rover Main Track View</h3>", unsafe_allow_html=True)
        frame_placeholder = st.empty()
        control_placeholder = st.empty()

    with col2:
        st.markdown("<h3>🔍 Target Diagnosis Engine</h3>", unsafe_allow_html=True)
        result_placeholder = st.container()

    if st.session_state.frozen_frame is not None:
        b64_display = get_image_base64(st.session_state.frozen_frame)

        if control_placeholder.button("🔄 Resume Stream Tracking", use_container_width=True):
            st.session_state.frozen_frame = None
            st.session_state.api_results = None
            st.rerun()

        if st.session_state.api_results:
            with col1:
                render_combined_scan_image(st.session_state.api_results, b64_display)
            with result_placeholder:
                render_diagnosis_panel(st.session_state.api_results, b64_display, show_image=False)
                render_ask_box(
                    st.session_state.api_results.get("species", "Unknown"),
                    st.session_state.api_results.get("condition", "Unknown"),
                    key_prefix="esp32"
                )
        else:
            frame_placeholder.markdown(f"""
                <div class="camera-box"><img src="{b64_display}" style="width:100%; border-radius:12px;" /></div>
            """, unsafe_allow_html=True)

    else:
        frame_placeholder.markdown(f"""
            <div class="camera-box">
                <img src="{STREAM_URL}" style="width:100%;" onerror="this.style.opacity=0.3;" />
            </div>
        """, unsafe_allow_html=True)

        if control_placeholder.button("📸 Lock Target & Scan Asset 🌿", use_container_width=True):
            with st.spinner("Grabbing snapshot from rover camera..."):
                try:
                    snap_resp = requests.get(CAPTURE_URL, timeout=5)
                except Exception as e:
                    st.error(f"Could not reach ESP32 at {CAPTURE_URL}: {e}")
                    snap_resp = None

            if snap_resp is not None and snap_resp.status_code == 200:
                img_array = np.frombuffer(snap_resp.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    st.error("Snapshot came back corrupted — try scanning again.")
                else:
                    with st.spinner("Executing Object Lock-on & MobileNet Diagnostics..."):
                        try:
                            lat, lng = get_current_gps()
                            _, encoded_img = cv2.imencode('.jpg', frame)
                            files = {"file": ("rover_snapshot.jpg", encoded_img.tobytes(), "image/jpeg")}
                            form_data = {}
                            if lat is not None:
                                form_data["latitude"] = str(lat)
                                form_data["longitude"] = str(lng)
                            response = requests.post(FASTAPI_URL, files=files, data=form_data, timeout=90)

                            if response.status_code == 200:
                                st.session_state.api_results = response.json()
                                st.session_state.frozen_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.rerun()
                            else:
                                st.error(f"API returned status {response.status_code}: {response.text[:200]}")
                        except Exception as e:
                            st.error(f"Rover connection protocol interrupted: {e}")
            elif snap_resp is not None:
                st.error(f"ESP32 snapshot endpoint returned status {snap_resp.status_code}.")

elif mode == "Upload Image Asset":
    st.subheader("📤 Manual Target Image Upload")
    uploaded_file = st.file_uploader("Upload farm asset file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([5, 4])
        file_bytes = uploaded_file.getvalue()
        opencv_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        if opencv_img is None:
            st.error("This file couldn't be read as an image. Please upload a valid JPG/PNG.")
        else:
            rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
            b64_display = get_image_base64(rgb_img)

            with col1:
                st.markdown("<h3>🎯 Uploaded Target Asset</h3>", unsafe_allow_html=True)
                if st.session_state.manual_api_results:
                    render_combined_scan_image(st.session_state.manual_api_results, b64_display)
                else:
                    st.markdown(f"""
                        <div class="camera-box"><img src="{b64_display}" style="width:100%;" /></div>
                    """, unsafe_allow_html=True)

                if st.button("Run Manual Asset Diagnostics", use_container_width=True):
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    with st.spinner("Analyzing manual entry asset..."):
                        try:
                            response = requests.post(FASTAPI_URL, files=files, timeout=90)
                            if response.status_code == 200:
                                st.session_state.manual_api_results = response.json()
                                st.rerun()
                            else:
                                st.error(f"API returned status {response.status_code}: {response.text[:200]}")
                        except Exception as e:
                            st.error(f"Could not connect to analysis pipeline: {e}")

            with col2:
                st.markdown("<h3>🔍 Diagnostics Evaluation</h3>", unsafe_allow_html=True)
                if st.session_state.manual_api_results:
                    render_diagnosis_panel(st.session_state.manual_api_results, b64_display, show_image=False)
                    render_ask_box(
                        st.session_state.manual_api_results.get("species", "Unknown"),
                        st.session_state.manual_api_results.get("condition", "Unknown"),
                        key_prefix="manual"
                    )
                else:
                    st.info("Click 'Run Manual Asset Diagnostics' to start the scan sequence.")

st.markdown("<div class='footer'>AgroVision AI Operations HUD v3.0 • Running on Autonomous Farm Rover Hardware</div>", unsafe_allow_html=True)