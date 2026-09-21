"""
Exuvia Counting Program - Main Streamlit App
Single unified program for camera, detection, training data, and analytics
"""
import streamlit as st
from pathlib import Path
import logging
import base64
import io
import re
import shutil
import socket
import subprocess
import zipfile
import time

# Import our modules
from camera import get_camera, reset_camera
from data_manager import get_data_manager
from tiler import ImageTiler

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== STREAMLIT PAGE CONFIG =====
st.set_page_config(
    page_title="Exuvia Counter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .big-number {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ===== DATA DIRECTORIES =====
DATA_DIR = Path("data")
CAPTURES_DIR = DATA_DIR / "captures"
TILES_DIR = DATA_DIR / "tiles"
DETECTIONS_DIR = DATA_DIR / "detections"
COUNTER_INPUTS_DIR = DATA_DIR / "counter_inputs"
MODELS_DIR = Path("models")
LAST_COUNTER_MODEL_FILE = DATA_DIR / "counter_last_model.txt"
COUNTER_SETTINGS_FILE = DATA_DIR / "counter_settings.json"

for d in [CAPTURES_DIR, TILES_DIR, DETECTIONS_DIR, COUNTER_INPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_available_model_options():
    """Return available local model files for the counter page."""
    candidates = []
    preferred_order = [
        "yolo26v5.pt",
        "yolo26v4.pt",
        "yolo26v3.pt",
        "yolo26v2.pt",
        "yolo26v1.pt",
        "yolov8.pt",
        "yolov11v1.pt",
    ]

    for name in preferred_order:
        path = MODELS_DIR / name
        if path.exists():
            candidates.append(path)

    for path in sorted(MODELS_DIR.glob("*.pt")):
        if path not in candidates:
            candidates.append(path)

    return {f"{path.stem} ({path.as_posix()})": str(path) for path in candidates}


def load_last_counter_model_path():
    """Load the last used counter model path from disk, if available."""
    try:
        if LAST_COUNTER_MODEL_FILE.exists():
            value = LAST_COUNTER_MODEL_FILE.read_text(encoding="utf-8").strip()
            return value or None
    except Exception:
        pass
    return None


def save_last_counter_model_path(model_path):
    """Persist the last used counter model path to disk."""
    try:
        LAST_COUNTER_MODEL_FILE.write_text(str(model_path).strip(), encoding="utf-8")
    except Exception:
        logger.warning("Unable to save last used counter model: %s", model_path)


DEFAULT_COUNTER_SETTINGS = {
    "confidence": 0.50,
    "iou": 0.45,
    "overlap": 80,
    "edge_margin_px": 8,
    "min_box_area_px": 24,
    "imgsz": 640,
}


def load_counter_settings():
    """Load persisted counter settings from disk, falling back to defaults."""
    import json
    try:
        if COUNTER_SETTINGS_FILE.exists():
            data = json.loads(COUNTER_SETTINGS_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_COUNTER_SETTINGS, **data}
    except Exception:
        pass
    return dict(DEFAULT_COUNTER_SETTINGS)


def save_counter_settings(settings_dict):
    """Persist counter settings to disk."""
    import json
    try:
        COUNTER_SETTINGS_FILE.write_text(
            json.dumps(settings_dict, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning("Unable to save counter settings.")


def apply_zoom_to_jpeg(jpeg_bytes, zoom):
    if jpeg_bytes is None or zoom <= 1:
        return jpeg_bytes

    import cv2
    import numpy as np

    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return jpeg_bytes

    height, width = image.shape[:2]
    crop_height = max(1, height // zoom)
    crop_width = max(1, width // zoom)
    y1 = (height - crop_height) // 2
    x1 = (width - crop_width) // 2
    cropped = image[y1:y1 + crop_height, x1:x1 + crop_width]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes() if ok else jpeg_bytes


def estimate_tile_count(width, height, tile_size, overlap):
    """Estimate tile count using the same stepping logic as ImageTiler."""
    stride = max(1, tile_size - overlap)
    coords = set()

    for y in range(0, max(1, height - tile_size + 1), stride):
        for x in range(0, max(1, width - tile_size + 1), stride):
            if x + tile_size <= width and y + tile_size <= height:
                coords.add((x, y))

    if width % stride != 0 and width >= tile_size:
        x = width - tile_size
        for y in range(0, max(1, height - tile_size + 1), stride):
            if y + tile_size <= height:
                coords.add((x, y))

    if height % stride != 0 and height >= tile_size:
        y = height - tile_size
        for x in range(0, max(1, width - tile_size + 1), stride):
            if x + tile_size <= width:
                coords.add((x, y))

    if width % stride != 0 and height % stride != 0 and width >= tile_size and height >= tile_size:
        coords.add((width - tile_size, height - tile_size))

    return len(coords)


def build_tiles_zip(tiles, coords, folder_name, image_id, tile_size=256):
    """Build a ZIP archive from generated tiles (+ coords.txt) and return bytes."""
    import cv2

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        # write coordinates manifest
        lines = ["tile_index,x,y,tile_size\n"]
        for i, (x, y) in enumerate(coords):
            lines.append(f"{i},{x},{y},{tile_size}\n")
        zipf.writestr(f"{folder_name}/{image_id}_coords.txt", "".join(lines))

        for i, (tile, (x, y)) in enumerate(zip(tiles, coords)):
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            filename = f"{image_id}_tile_{i:03d}_x{x}_y{y}.jpg"
            zipf.writestr(f"{folder_name}/{filename}", enc.tobytes())
    return buffer.getvalue()


def get_current_request_url(default_port=8501):
    """Return the URL currently used by the browser when available."""
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None)
    if not headers:
        return None

    host = headers.get("x-forwarded-host") or headers.get("host")
    if not host:
        return None

    scheme = headers.get("x-forwarded-proto") or "http"
    if ":" not in host and default_port:
        host = f"{host}:{default_port}"

    return f"{scheme}://{host}"


def get_primary_lan_ip():
    """Best-effort LAN IP for same-network access."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        for ip in hostname_ips:
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return None


def get_tailscale_ip():
    """Return the first Tailscale IPv4 when available."""
    if shutil.which("tailscale") is None:
        return None

    try:
        completed = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0:
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass

    return None


def get_funnel_url():
    """Return the public Funnel URL when enabled."""
    if shutil.which("tailscale") is None:
        return None

    try:
        completed = subprocess.run(
            ["tailscale", "funnel", "status"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0:
            match = re.search(r"https://\S+", completed.stdout)
            if match:
                return match.group(0).rstrip(",")
    except Exception:
        pass

    return None


def get_access_links(default_port=8501):
    """Collect the URLs that can be used to reach this app."""
    links = []
    seen = set()

    def add_link(label, url):
        if url and url not in seen:
            links.append((label, url))
            seen.add(url)

    add_link("Current browser URL", get_current_request_url(default_port=default_port))
    add_link("Local on this Pi", f"http://localhost:{default_port}")

    lan_ip = get_primary_lan_ip()
    if lan_ip:
        add_link("LAN / same network", f"http://{lan_ip}:{default_port}")

    tailscale_ip = get_tailscale_ip()
    if tailscale_ip:
        add_link("Tailscale", f"http://{tailscale_ip}:{default_port}")

    add_link("Public Funnel", get_funnel_url())
    return links


def get_default_quad_points(width, height, margin_ratio=0.08):
    """Return default clockwise tray quadrilateral points for a new image."""
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    return {
        "tl": [float(margin_x), float(margin_y)],
        "tr": [float(width - margin_x), float(margin_y)],
        "br": [float(width - margin_x), float(height - margin_y)],
        "bl": [float(margin_x), float(height - margin_y)],
    }


def scale_quad_points(quad_points, scale_x, scale_y):
    """Scale quadrilateral point coordinates by x/y factors."""
    return {
        key: [float(val[0]) * float(scale_x), float(val[1]) * float(scale_y)]
        for key, val in quad_points.items()
    }


def build_quad_canvas_drawing(quad_points, radius=10):
    """Create drawable-canvas initial JSON with 4 draggable corner circles."""
    corner_colors = {
        "tl": "#ff4f4f",
        "tr": "#52c41a",
        "br": "#4096ff",
        "bl": "#f0c000",
    }
    objects = []

    for key in ("tl", "tr", "br", "bl"):
        x, y = quad_points[key]
        objects.append(
            {
                "type": "circle",
                "left": float(x) - radius,
                "top": float(y) - radius,
                "radius": radius,
                "fill": corner_colors[key],
                "stroke": "#ffffff",
                "strokeWidth": 2,
                "originX": "left",
                "originY": "top",
                "corner": key,
                "name": key,
                "hasControls": False,
                "hasBorders": False,
                "lockScalingX": True,
                "lockScalingY": True,
                "lockRotation": True,
                "lockSkewingX": True,
                "lockSkewingY": True,
            }
        )

    return {"version": "4.4.0", "objects": objects}


def quad_points_from_canvas_json(json_data, fallback_quad, max_w, max_h):
    """Extract tl/tr/br/bl points from drawable-canvas JSON circle objects."""
    import math

    if not json_data or "objects" not in json_data:
        return fallback_quad

    points = {k: [float(v[0]), float(v[1])] for k, v in fallback_quad.items()}

    def _center_from_obj(obj):
        radius = float(obj.get("radius", 10.0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        cx = left + radius * scale_x
        cy = top + radius * scale_y
        cx = min(max(cx, 0.0), float(max_w - 1))
        cy = min(max(cy, 0.0), float(max_h - 1))
        return [cx, cy]

    color_to_key = {
        "#ff4f4f": "tl",
        "#52c41a": "tr",
        "#4096ff": "br",
        "#f0c000": "bl",
    }

    keyed_hits = set()
    circle_centers = []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "circle":
            continue

        center = _center_from_obj(obj)
        circle_centers.append(center)

        key = None
        fill = str(obj.get("fill", "")).lower()
        if fill in color_to_key:
            key = color_to_key[fill]
        if key is None:
            key = obj.get("corner") or obj.get("name")
        if key in points:
            points[key] = center
            keyed_hits.add(key)

    # Some canvas transforms can drop custom metadata keys.
    # If that happens, re-attach circles to corners by nearest fallback point.
    missing_keys = [k for k in ("tl", "tr", "br", "bl") if k not in keyed_hits]
    if missing_keys and circle_centers:
        remaining_centers = circle_centers.copy()
        for key in missing_keys:
            fx, fy = fallback_quad[key]
            best_idx = 0
            best_dist = float("inf")
            for idx, (cx, cy) in enumerate(remaining_centers):
                d = math.hypot(cx - fx, cy - fy)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            points[key] = remaining_centers.pop(best_idx)
            if not remaining_centers:
                break

    return points


def extract_click_points_from_canvas_json(json_data, max_w, max_h, max_points=4):
    """Extract clicked point centers from drawable-canvas JSON circle objects."""
    if not json_data or "objects" not in json_data:
        return []

    points = []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "circle":
            continue

        radius = float(obj.get("radius", 0.0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))

        cx = left + radius * scale_x
        cy = top + radius * scale_y
        cx = min(max(cx, 0.0), float(max_w - 1))
        cy = min(max(cy, 0.0), float(max_h - 1))

        points.append([cx, cy])
        if len(points) >= int(max_points):
            break

    return points


def quad_points_from_click_points(click_points):
    """Convert any 4 click points into tl/tr/br/bl point dict."""
    import numpy as np

    if click_points is None or len(click_points) < 4:
        return None

    pts = np.array(click_points[:4], dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return {
        "tl": [float(ordered[0][0]), float(ordered[0][1])],
        "tr": [float(ordered[1][0]), float(ordered[1][1])],
        "br": [float(ordered[2][0]), float(ordered[2][1])],
        "bl": [float(ordered[3][0]), float(ordered[3][1])],
    }


def order_quad_points(quad_points):
    """Order 4 points as tl, tr, br, bl for perspective transform."""
    import numpy as np

    pts = np.array([
        quad_points["tl"],
        quad_points["tr"],
        quad_points["br"],
        quad_points["bl"],
    ], dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def draw_quad_overlay(image_rgb, quad_points):
    """Draw the tray quadrilateral and corner markers over an RGB image."""
    import cv2
    import numpy as np

    overlay = image_rgb.copy()
    pts = order_quad_points(quad_points).astype(np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 180, 0), thickness=4)

    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(pts):
        cv2.circle(overlay, (int(x), int(y)), 9, (255, 80, 80), -1)
        cv2.putText(
            overlay,
            labels[i],
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def warp_tray_from_quad(image_bgr, quad_points, upscale_factor=1.4):
    """Perspective-warp image using tray quadrilateral and optional upscale factor."""
    import cv2
    import numpy as np

    rect = order_quad_points(quad_points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_w = max(16, int(max(width_a, width_b)))
    max_h = max(16, int(max(height_a, height_b)))

    out_w = max(64, int(max_w * float(upscale_factor)))
    out_h = max(64, int(max_h * float(upscale_factor)))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped_bgr = cv2.warpPerspective(image_bgr, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)


# ===== SIDEBAR =====
st.sidebar.title("🐛 Exuvia Counter")
st.sidebar.markdown("---")

params = st.query_params
page = params.get("page", "camera-focus")

st.sidebar.markdown("### Menu")

if st.sidebar.button("Video Capture (Focus)", use_container_width=True):
    st.query_params["page"] = "camera-focus"
    st.rerun()
if st.sidebar.button("Image Capture", use_container_width=True):
    st.query_params["page"] = "image-capture"
    st.rerun()
if st.sidebar.button("Exuvia Counter", use_container_width=True):
    st.query_params["page"] = "exuvia-counter"
    st.rerun()
if st.sidebar.button("Data & Tables", use_container_width=True):
    st.query_params["page"] = "data-tables"
    st.rerun()
if st.sidebar.button("Info", use_container_width=True):
    st.query_params["page"] = "technical-info"
    st.rerun()

access_links = get_access_links()
if access_links:
    st.sidebar.markdown("### Links")
    for label, url in access_links:
        st.sidebar.markdown(f"**{label}:**")
        st.sidebar.code(url, language=None)

# ----- shared camera singleton (used by both pages) -----
camera = get_camera(use_pi_camera=True)
data_mgr = get_data_manager()

ZOOM_PROFILES = {
    1: {"resolution": (1920, 1440), "fps": 8},
    2: {"resolution": (1920, 1440), "fps": 8},
    3: {"resolution": (2028, 1520), "fps": 5},
}

# per-page session state defaults
if "live_feed_on" not in st.session_state:
    st.session_state.live_feed_on = False          # feed OFF by default
if "current_zoom" not in st.session_state:
    st.session_state.current_zoom = 1
if "capture_preview_jpeg" not in st.session_state:
    st.session_state.capture_preview_jpeg = None   # bytes of last captured snapshot
if "capture_preview_path" not in st.session_state:
    st.session_state.capture_preview_path = None   # temp Path object
if "tiles_zip_bytes" not in st.session_state:
    st.session_state.tiles_zip_bytes = None
if "tiles_zip_name" not in st.session_state:
    st.session_state.tiles_zip_name = None
if "counter_preview_jpeg" not in st.session_state:
    st.session_state.counter_preview_jpeg = None
if "counter_preview_path" not in st.session_state:
    st.session_state.counter_preview_path = None
if "counter_last_result" not in st.session_state:
    st.session_state.counter_last_result = None
if "counter_result_logged" not in st.session_state:
    st.session_state.counter_result_logged = False
if "focus_wait_started_at" not in st.session_state:
    st.session_state.focus_wait_started_at = None
if "focus_restart_attempted" not in st.session_state:
    st.session_state.focus_restart_attempted = False
if "counter_batch_index" not in st.session_state:
    st.session_state.counter_batch_index = 0
if "counter_last_upload_token" not in st.session_state:
    st.session_state.counter_last_upload_token = None
if "counter_show_upload_picker" not in st.session_state:
    st.session_state.counter_show_upload_picker = False
if "counter_batch_number" not in st.session_state:
    st.session_state.counter_batch_number = 1
if "counter_batch_data" not in st.session_state:
    st.session_state.counter_batch_data = []
if "counter_quad_points" not in st.session_state:
    st.session_state.counter_quad_points = None
if "counter_quad_source" not in st.session_state:
    st.session_state.counter_quad_source = None
if "counter_canvas_seed_token" not in st.session_state:
    st.session_state.counter_canvas_seed_token = None
if "counter_canvas_seed_drawing" not in st.session_state:
    st.session_state.counter_canvas_seed_drawing = None
if "counter_sweep_results" not in st.session_state:
    st.session_state.counter_sweep_results = []


# ===========================
# PAGE 1 – CAMERA FOCUS
# ===========================
if page == "camera-focus":
    st.title("🐛 Video Capture (Focus)")
    st.caption(
        "The Pi HQ camera has a manual lens. "
        "Use the live feed below to adjust focus. "
        "Refer to the [Guide/Manual](https://docs.google.com/document/d/1wDBeD1E9GZ5ZaHm4I_H8D6K9ydvff_BBw3AuetpO-Ec/edit?usp=sharing) "
        "for info on adjusting the three focus rings."
    )

    # --- top controls ---
    ctrl_cols = st.columns([1, 1, 6])
    with ctrl_cols[0]:
        if st.session_state.live_feed_on:
            if st.button("Stop Live Feed"):
                st.session_state.live_feed_on = False
                camera.stop_preview_stream()
                st.rerun()
        else:
            if st.button("Start Live Feed"):
                st.session_state.live_feed_on = True
                st.session_state.focus_wait_started_at = time.time()
                st.session_state.focus_restart_attempted = False
                st.rerun()

    with ctrl_cols[1]:
        if st.button("Reconnect Camera"):
            with st.spinner("Reinitialising..."):
                reset_camera(use_pi_camera=True)
                st.session_state.live_feed_on = False
                st.session_state.focus_wait_started_at = None
                st.session_state.focus_restart_attempted = False
            st.rerun()

    camera_status = camera.get_status() if hasattr(camera, "get_status") else {}

    if not camera.is_available():
        st.error("Camera not available. Click Reconnect Camera.")
        if camera_status.get("last_error"):
            st.caption(f"Last error: {camera_status['last_error']}")
    else:
        # --- zoom buttons ---
        zoom_cols = st.columns([1, 1, 1, 9])
        with zoom_cols[0]:
            if st.button("1x", key="btn_1x"):
                st.session_state.current_zoom = 1
        with zoom_cols[1]:
            if st.button("2x", key="btn_2x"):
                st.session_state.current_zoom = 2
        with zoom_cols[2]:
            if st.button("3x", key="btn_3x"):
                st.session_state.current_zoom = 3

        # make sure the right stream resolution is running
        if st.session_state.live_feed_on and camera.camera_type == "rpicam_cli":
            profile = ZOOM_PROFILES[st.session_state.current_zoom]
            camera.start_preview_stream(fps=profile["fps"], resolution=profile["resolution"])

        if not st.session_state.live_feed_on:
            st.info("Live feed is off. Click **Start Live Feed** to begin.")
        else:
            @st.fragment(run_every="350ms")
            def render_focus_preview():
                zoom = st.session_state.current_zoom
                jpeg_bytes = camera.get_preview_jpeg()
                if jpeg_bytes is not None:
                    st.session_state["_last_preview_jpeg"] = jpeg_bytes
                    st.session_state.focus_wait_started_at = None
                    st.session_state.focus_restart_attempted = False
                else:
                    jpeg_bytes = st.session_state.get("_last_preview_jpeg")

                display_jpeg = apply_zoom_to_jpeg(jpeg_bytes, zoom)
                if display_jpeg is not None:
                    b64 = base64.b64encode(display_jpeg).decode()
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{b64}" '
                        f'style="width:100%;height:auto;border-radius:4px;display:block;">',
                        unsafe_allow_html=True,
                    )
                    status = camera.get_status()
                    pcfg = status.get("preview_config") or {}
                    res = pcfg.get("resolution", ("?", "?"))
                    fps_t = pcfg.get("fps", "?")
                    st.caption(f"Zoom: {zoom}x | {res[0]}x{res[1]} @ {fps_t} fps")
                else:
                    if st.session_state.focus_wait_started_at is None:
                        st.session_state.focus_wait_started_at = time.time()

                    waited_s = time.time() - st.session_state.focus_wait_started_at
                    st.markdown(
                        """
                        <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
                          <div style="width:16px;height:16px;border:2px solid #d0d0d0;border-top:2px solid #2e7d32;border-radius:50%;animation:spin 0.9s linear infinite;"></div>
                          <span>Loading live preview...</span>
                        </div>
                        <style>
                          @keyframes spin {from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Recovery: if first frame stalls, restart preview process once.
                    if (
                        camera.camera_type == "rpicam_cli"
                        and waited_s > 4.0
                        and not st.session_state.focus_restart_attempted
                    ):
                        profile = ZOOM_PROFILES.get(st.session_state.current_zoom, ZOOM_PROFILES[1])
                        camera.stop_preview_stream()
                        camera.start_preview_stream(fps=profile["fps"], resolution=profile["resolution"])
                        st.session_state.focus_restart_attempted = True
                        st.caption("Preview startup was slow; auto-retrying once...")

            render_focus_preview()


# ===========================
# PAGE 2 – IMAGE CAPTURE
# ===========================
elif page == "image-capture":
    import shutil as _shutil

    st.title("🐛 Image Capture")
    st.caption("Take high resolution photos of Exuvia trays here. Images can also be tiled/processed for training CV below.")

    # Stop the live feed if it was running from Camera Focus
    if camera.is_preview_running():
        camera.stop_preview_stream()
        st.session_state.live_feed_on = False

    camera_status = camera.get_status() if hasattr(camera, "get_status") else {}

    if not camera.is_available():
        st.error("Camera not available. Go to Camera Focus and click Reconnect Camera.")
        if camera_status.get("last_error"):
            st.caption(f"Last error: {camera_status['last_error']}")
    else:
        take_btn = st.button("Take Photo")
        tray_id = st.text_input("Tray ID", value="tray_001")
        notes   = st.text_area("Notes", height=80)

        if take_btn:
            # discard any previous unsaved temp file
            prev = st.session_state.capture_preview_path
            if prev and Path(prev).exists():
                try:
                    Path(prev).unlink()
                except Exception:
                    pass
            st.session_state.capture_preview_jpeg = None
            st.session_state.capture_preview_path = None
            st.session_state.tiles_zip_bytes = None
            st.session_state.tiles_zip_name = None

            with st.spinner("Capturing full-resolution image..."):
                tmp_path = camera.capture_tray_jpeg()

            if tmp_path is None:
                st.error(f"Capture failed: {camera_status.get('last_error', 'unknown error')}")
            else:
                # Read and compress for preview display
                import cv2 as _cv2
                img = _cv2.imread(str(tmp_path))
                if img is not None:
                    ok, enc = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                    st.session_state.capture_preview_jpeg = enc.tobytes() if ok else None
                st.session_state.capture_preview_path = str(tmp_path)

        # Show preview + save/discard controls if we have a pending capture
        if st.session_state.capture_preview_jpeg:
            st.subheader("Preview")
            b64 = base64.b64encode(st.session_state.capture_preview_jpeg).decode()
            st.markdown(
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:100%;height:auto;border-radius:4px;display:block;">',
                unsafe_allow_html=True,
            )

            tmp_path = Path(st.session_state.capture_preview_path)
            st.caption(f"Temp file: {tmp_path}")

            try:
                preview_data = tmp_path.read_bytes()
            except Exception:
                preview_data = None

            preview_h = preview_w = None
            try:
                import cv2 as _cv2
                _img_meta = _cv2.imread(str(tmp_path))
                if _img_meta is not None:
                    preview_h, preview_w = _img_meta.shape[:2]
            except Exception:
                pass

            if preview_data is not None:
                st.caption(f"Preview source size: {len(preview_data) / (1024 * 1024):.2f} MB")
            if preview_w and preview_h:
                st.caption(f"Preview source resolution: {preview_w}x{preview_h} px")

            if preview_data:
                st.download_button(
                    "Save Image",
                    data=preview_data,
                    file_name=tmp_path.name,
                    mime="image/jpeg",
                    help="Downloads this image to the computer currently using the web UI.",
                )

            save_col, discard_col = st.columns([1, 1])
            with save_col:
                save_btn = st.button("Save Image to Pi", type="primary")
            with discard_col:
                discard_btn = st.button("Discard")

            if save_btn:
                CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
                from datetime import datetime as _dt
                timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
                dest = CAPTURES_DIR / f"{tray_id}_{timestamp}_full.jpg"
                _shutil.copy2(str(tmp_path), str(dest))
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
                st.session_state.capture_preview_jpeg = None
                st.session_state.capture_preview_path = None
                data_mgr.add_detection(
                    tray_id=tray_id,
                    zoom=1,
                    model="IMAGE_CAPTURE",
                    count=0,
                    mean_confidence=0.0,
                    image_path=dest,
                    notes=notes,
                )
                st.success(f"Saved to: {dest}")
                st.info(f"Full path: {dest.resolve()}")

            st.markdown("---")
            st.subheader("Tile Processing")
            st.caption("Roboflow recommended settings: 640×640 px tiles at 80 px (12.5%) overlap.")

            tile_size = st.number_input("Tile size (pixels)", min_value=128, max_value=1024, value=640, step=32)
            overlap = st.number_input("Overlap (pixels)", min_value=0, max_value=512, value=80, step=10)
            if tile_size > 0:
                overlap_pct = (int(overlap) / int(tile_size)) * 100
                st.caption(f"Overlap: {int(overlap)} px = {overlap_pct:.1f}% of tile size")

            if preview_w and preview_h:
                estimated_tiles = estimate_tile_count(
                    width=preview_w,
                    height=preview_h,
                    tile_size=int(tile_size),
                    overlap=int(overlap),
                )
                st.caption(f"Estimated tiles: {estimated_tiles}")

            tile_pi_col, tile_zip_col = st.columns([1, 1])
            save_computer_btn = False
            with tile_pi_col:
                save_tiles_pi_btn = st.button("Save Tiles to Pi")
            with tile_zip_col:
                if st.session_state.tiles_zip_bytes is None:
                    save_computer_btn = st.button("Save Tiles to Computer", use_container_width=True)
                else:
                    st.success("Tiles ready.")
                    st.download_button(
                        "Download ZIP",
                        data=st.session_state.tiles_zip_bytes,
                        file_name=st.session_state.tiles_zip_name or "tiles.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary",
                    )

            if save_tiles_pi_btn or save_computer_btn:
                tiles, coords = ImageTiler.create_tiles(tmp_path, tile_size=int(tile_size), overlap=int(overlap))
                if not tiles:
                    st.error("No tiles were generated from the current image.")
                else:
                    tile_folder = f"{tray_id}_({int(tile_size)}px_o{int(overlap)}px)"

                    if save_tiles_pi_btn:
                        saved_tiles = ImageTiler.save_tiles(
                            tiles,
                            coords,
                            output_dir=TILES_DIR,
                            tray_id=tile_folder,
                            image_id=tmp_path.stem,
                        )
                        # write coordinates manifest alongside the tiles
                        coords_path = TILES_DIR / tile_folder / f"{tmp_path.stem}_coords.txt"
                        with open(coords_path, "w") as _cf:
                            _cf.write("tile_index,x,y,tile_size\n")
                            for _i, (_x, _y) in enumerate(coords):
                                _cf.write(f"{_i},{_x},{_y},{int(tile_size)}\n")
                        st.success(f"Saved {len(saved_tiles)} tiles + coords.txt to: {TILES_DIR / tile_folder}")

                    if save_computer_btn:
                        with st.spinner("Generating tiles..."):
                            st.session_state.tiles_zip_bytes = build_tiles_zip(
                                tiles,
                                coords,
                                folder_name=tile_folder,
                                image_id=tmp_path.stem,
                                tile_size=int(tile_size),
                            )
                            st.session_state.tiles_zip_name = f"{tile_folder}.zip"
                        st.rerun()

            if discard_btn:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
                st.session_state.capture_preview_jpeg = None
                st.session_state.capture_preview_path = None
                st.session_state.tiles_zip_bytes = None
                st.session_state.tiles_zip_name = None
                st.rerun()

elif page == "exuvia-counter":
    st.title("🐛 Exuvia Counter")
    st.caption("Capture a tray image, then process it with the selected model.")
    batch_col, reset_col = st.columns([3, 1])
    with batch_col:
        st.info(f"Batch {st.session_state.counter_batch_number} — Photo {max(1, int(st.session_state.counter_batch_index))}")
    with reset_col:
        if st.button("New Batch", use_container_width=True, key="counter_reset_batch"):
            st.session_state.counter_batch_number += 1
            st.session_state.counter_batch_data = []
            st.session_state.counter_batch_index = 0
            st.session_state.counter_preview_jpeg = None
            st.session_state.counter_preview_path = None
            st.session_state.counter_quad_points = None
            st.session_state.counter_quad_source = None
            st.session_state.counter_canvas_seed_token = None
            st.session_state.counter_canvas_seed_drawing = None
            st.session_state.counter_last_result = None
            st.session_state.counter_result_logged = False
            st.session_state.counter_last_upload_token = None
            st.rerun()

    # Keep focus preview from consuming camera resources while counting.
    if camera.is_preview_running():
        camera.stop_preview_stream()
        st.session_state.live_feed_on = False

    model_options = get_available_model_options()
    model_label = next(iter(model_options), None)
    last_model_path = load_last_counter_model_path()
    if model_options:
        if last_model_path:
            for label, path in model_options.items():
                if Path(path).as_posix() == Path(last_model_path).as_posix():
                    model_label = label
                    break
        if model_label is None:
            for hint in ("yolo26v5", "yolo26v4", "yolo26v3", "yolo26v2", "yolo26", "yolov8", "yolov11"):
                matched = [lbl for lbl in model_options.keys() if hint in lbl.lower()]
                if matched:
                    model_label = matched[0]
                    break
    model_path = Path(model_options[model_label]) if model_label else None

    _saved_settings = load_counter_settings()

    # Seed slider session_state keys from disk on first load in this session.
    if "counter_settings_initialized" not in st.session_state:
        st.session_state.counter_confidence = float(_saved_settings["confidence"])
        st.session_state.counter_iou = float(_saved_settings["iou"])
        st.session_state.counter_overlap = int(_saved_settings["overlap"])
        st.session_state.counter_edge_margin = int(_saved_settings["edge_margin_px"])
        st.session_state.counter_min_box_area = int(_saved_settings["min_box_area_px"])
        st.session_state.counter_model_imgsz = int(_saved_settings["imgsz"])
        st.session_state.counter_settings_initialized = True

    defaults = {
        "confidence": float(_saved_settings["confidence"]),
        "iou": float(_saved_settings["iou"]),
        "tile_size": int(_saved_settings["imgsz"]),
        "overlap": int(_saved_settings["overlap"]),
        "edge_margin_px": int(_saved_settings["edge_margin_px"]),
        "min_box_area_px": int(_saved_settings["min_box_area_px"]),
    }
    confidence = defaults["confidence"]
    iou = defaults["iou"]
    tile_size = defaults["tile_size"]
    overlap = defaults["overlap"]

    if not model_options:
        st.error("No model files were found in the models folder.")
        st.info("Copy your trained .pt model into the models folder and retry.")
    else:
        source_col1, source_col2, source_col3 = st.columns([2, 1, 1])
        with source_col1:
            take_counter_btn = st.button("Take Counter Photo", type="primary", use_container_width=True, key="counter_take_photo")
        with source_col2:
            use_saved_pi_btn = st.button("Use Saved Pi Photo", use_container_width=True, key="counter_use_saved")
        with source_col3:
            open_upload_btn = st.button("Upload Photo", use_container_width=True, key="counter_open_upload")

        if open_upload_btn:
            st.session_state.counter_show_upload_picker = True

        uploaded_counter_file = None
        if st.session_state.counter_show_upload_picker:
            uploaded_counter_file = st.file_uploader(
                "Choose uploaded image",
                type=["jpg", "jpeg", "png"],
                help="Upload an external image for counting.",
                key="counter_upload_photo",
            )


        if take_counter_btn:
            prev = st.session_state.counter_preview_path
            if prev and Path(prev).exists():
                try:
                    Path(prev).unlink()
                except Exception:
                    pass
            st.session_state.counter_preview_jpeg = None
            st.session_state.counter_preview_path = None
            st.session_state.counter_last_result = None
            st.session_state.counter_result_logged = False
            st.session_state.counter_sweep_results = []

            with st.spinner("Capturing full-resolution image for counting..."):
                tmp_path = camera.capture_tray_jpeg()

            if tmp_path is None:
                st.error("Capture failed. Try reconnecting camera on Video Capture (Focus).")
            else:
                import cv2 as _cv2
                img = _cv2.imread(str(tmp_path))
                if img is not None:
                    ok, enc = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                    st.session_state.counter_preview_jpeg = enc.tobytes() if ok else None
                st.session_state.counter_preview_path = str(tmp_path)
                st.session_state.counter_quad_points = None
                st.session_state.counter_quad_source = None
                st.session_state.counter_canvas_seed_token = None
                st.session_state.counter_canvas_seed_drawing = None
                st.session_state.counter_batch_index += 1
                st.success("Counter photo captured.")

        captures = sorted(CAPTURES_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if captures:
            selected_saved_name = st.selectbox(
                "Saved Pi photo",
                options=[p.name for p in captures],
                index=0,
                help="Choose a previously saved Pi capture.",
            )
            if use_saved_pi_btn:
                selected_saved_path = next(p for p in captures if p.name == selected_saved_name)
                import cv2 as _cv2
                img = _cv2.imread(str(selected_saved_path))
                if img is None:
                    st.error("Failed to read selected saved image.")
                else:
                    ok, enc = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                    st.session_state.counter_preview_jpeg = enc.tobytes() if ok else None
                    st.session_state.counter_preview_path = str(selected_saved_path)
                    st.session_state.counter_quad_points = None
                    st.session_state.counter_quad_source = None
                    st.session_state.counter_canvas_seed_token = None
                    st.session_state.counter_canvas_seed_drawing = None
                    st.session_state.counter_last_result = None
                    st.session_state.counter_result_logged = False
                    st.session_state.counter_sweep_results = []
                    st.session_state.counter_batch_index += 1
                    st.success(f"Loaded saved image: {selected_saved_path.name}")
        else:
            st.caption("No saved Pi captures found yet.")

        if uploaded_counter_file is not None:
            upload_bytes = uploaded_counter_file.getvalue()
            upload_token = f"{uploaded_counter_file.name}:{len(upload_bytes)}"
            if upload_token != st.session_state.counter_last_upload_token:
                upload_name = Path(uploaded_counter_file.name).stem
                upload_ext = Path(uploaded_counter_file.name).suffix.lower() or ".jpg"
                upload_path = COUNTER_INPUTS_DIR / f"upload_{upload_name}{upload_ext}"
                upload_path.write_bytes(upload_bytes)

                import cv2 as _cv2
                img = _cv2.imread(str(upload_path))
                if img is None:
                    st.error("Uploaded file could not be decoded as an image.")
                else:
                    ok, enc = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                    st.session_state.counter_preview_jpeg = enc.tobytes() if ok else None
                    st.session_state.counter_preview_path = str(upload_path)
                    st.session_state.counter_quad_points = None
                    st.session_state.counter_quad_source = None
                    st.session_state.counter_canvas_seed_token = None
                    st.session_state.counter_canvas_seed_drawing = None
                    st.session_state.counter_last_result = None
                    st.session_state.counter_result_logged = False
                    st.session_state.counter_sweep_results = []
                    st.session_state.counter_batch_index += 1
                    st.session_state.counter_last_upload_token = upload_token
                    st.success(f"Uploaded image ready: {upload_path.name}")
                    st.session_state.counter_show_upload_picker = False

        if st.session_state.counter_preview_jpeg and st.session_state.counter_preview_path:
            st.subheader("Counter Preview")
            counter_img_path = Path(st.session_state.counter_preview_path)
            st.caption(f"Source: {counter_img_path.name}")
            st.caption("No point selection needed. YOLO runs directly on this image.")

            import cv2 as _cv2

            source_bgr = _cv2.imread(str(counter_img_path))
            if source_bgr is None:
                st.error("Could not read source image for preview.")
                source_rgb = None
            else:
                source_rgb = _cv2.cvtColor(source_bgr, _cv2.COLOR_BGR2RGB)

            if source_rgb is not None:
                st.image(source_rgb, caption="Counter source image", use_container_width=True)

            st.markdown("---")
            st.subheader("Processing Setup")
            with st.form("counter_process_form"):
                model_keys = list(model_options.keys())
                default_model_index = model_keys.index(model_label) if model_label in model_keys else 0
                model_label = st.selectbox("Model option", options=model_keys, index=default_model_index, key="counter_model_option")
                model_path = Path(model_options[model_label])
                st.caption(f"Model path: {model_path}")

                setup_left, setup_right = st.columns([2, 1])
                with setup_left:
                    st.info("Pipeline: overlap tiling + global merge (NMS/dedupe) for more stable seam handling.")
                with setup_right:
                    with st.expander("Advanced", expanded=False):
                        st.caption("Confidence: minimum score for a detection to be kept.")
                        confidence = st.slider("Confidence", min_value=0.05, max_value=0.95, value=defaults["confidence"], step=0.05, key="counter_confidence")
                        st.caption("IoU threshold used by YOLO post-processing.")
                        iou = st.slider("IoU", min_value=0.10, max_value=0.95, value=defaults["iou"], step=0.05, key="counter_iou")
                        st.caption("Tile overlap reduces split-object misses at tile boundaries.")
                        overlap = st.slider("Overlap (px)", min_value=0, max_value=320, value=defaults["overlap"], step=10, key="counter_overlap")
                        st.caption("Post-merge filter: ignore detections too close to image border.")
                        edge_margin_px = st.number_input("Edge margin (px)", min_value=0, max_value=64, value=defaults["edge_margin_px"], step=1, key="counter_edge_margin")
                        st.caption("Post-merge filter: ignore very small boxes (texture noise).")
                        min_box_area_px = st.number_input("Minimum box area (px^2)", min_value=0, max_value=2000, value=defaults["min_box_area_px"], step=4, key="counter_min_box_area")
                        st.caption("Per-tile YOLO input size.")
                        imgsz = st.number_input("Model input size", min_value=320, max_value=1280, value=defaults["tile_size"], step=32, key="counter_model_imgsz")

                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    process_btn = st.form_submit_button("Process Image", type="primary", use_container_width=True)
                with action_col2:
                    sweep_btn = st.form_submit_button("Run Threshold Sweep", use_container_width=True)
                with action_col3:
                    set_defaults_btn = st.form_submit_button("Set as Default", use_container_width=True)

                tile_size = int(imgsz)
                overlap = int(overlap)

            discard_counter_btn = st.button(
                "Discard Counter Image",
                use_container_width=True,
                key="counter_discard",
            )

            if set_defaults_btn:
                save_counter_settings({
                    "confidence": float(confidence),
                    "iou": float(iou),
                    "overlap": int(overlap),
                    "edge_margin_px": int(edge_margin_px),
                    "min_box_area_px": float(min_box_area_px),
                    "imgsz": int(imgsz),
                })
                st.success(f"Default settings saved: conf={float(confidence):.2f}, iou={float(iou):.2f}, overlap={int(overlap)}px, tile={int(imgsz)}px")

            if process_btn:
                from detector import get_detector
                save_last_counter_model_path(model_path)

                with st.spinner("Processing image with YOLO. Please wait 2-5 minutes..."):
                    detector = get_detector(str(model_path))
                    if not detector.is_available():
                        st.error("Detector unavailable. Ensure ultralytics is installed.")
                    elif source_rgb is None:
                        st.error("Source image is unavailable.")
                    else:
                        try:
                            detection = detector.detect_tiled(
                                image_source=str(counter_img_path),
                                tile_size=int(imgsz),
                                overlap=int(overlap),
                                confidence=float(confidence),
                                iou=float(iou),
                                edge_margin_px=int(edge_margin_px),
                                min_box_area_px=float(min_box_area_px),
                            )
                        except Exception as exc:
                            logger.exception("Counter processing crashed")
                            detection = None
                            st.error(f"Processing crashed: {exc}")

                        if detection is None:
                            st.error("Processing failed before producing output. Try again with lower model input size (for example 512 or 384).")

                        if detection and detection.get("image_with_boxes") is not None:
                            from datetime import datetime as _dt
                            out_name = f"{counter_img_path.stem}_tiled_detected_{_dt.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            out_path = DETECTIONS_DIR / out_name
                            out_bgr = _cv2.cvtColor(detection["image_with_boxes"], _cv2.COLOR_RGB2BGR)
                            _cv2.imwrite(str(out_path), out_bgr)
                            detection["saved_path"] = str(out_path)
                            detection["model_name"] = model_path.name
                            detection["settings"] = {
                                "mode": "tiled_overlap_global_merge",
                                "confidence": float(confidence),
                                "iou": float(iou),
                                "imgsz": int(imgsz),
                                "overlap": int(overlap),
                                "edge_margin_px": int(edge_margin_px),
                                "min_box_area_px": float(min_box_area_px),
                            }
                        elif detection is not None:
                            st.warning("Processing completed but no annotated output image was produced.")

                        st.session_state.counter_last_result = detection
                        st.session_state.counter_result_logged = False

            if sweep_btn:
                from detector import get_detector
                save_last_counter_model_path(model_path)

                with st.spinner("Running threshold sweep on current image..."):
                    detector = get_detector(str(model_path))
                    sweep_iou = 0.45
                    sweep_conf_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
                    sweep_rows = []

                    if not detector.is_available():
                        st.error("Detector unavailable. Ensure ultralytics is installed.")
                    else:
                        for sweep_conf in sweep_conf_values:
                            started = time.perf_counter()
                            sweep_result = detector.detect_tiled(
                                image_source=str(counter_img_path),
                                tile_size=int(imgsz),
                                overlap=int(overlap),
                                confidence=float(sweep_conf),
                                iou=float(sweep_iou),
                                edge_margin_px=int(edge_margin_px),
                                min_box_area_px=float(min_box_area_px),
                            )
                            runtime_ms = (time.perf_counter() - started) * 1000.0

                            if sweep_result is None:
                                sweep_rows.append(
                                    {
                                        "conf": float(sweep_conf),
                                        "count": None,
                                        "mean_confidence": None,
                                        "runtime_ms": round(runtime_ms, 1),
                                    }
                                )
                            else:
                                sweep_rows.append(
                                    {
                                        "conf": float(sweep_conf),
                                        "count": int(sweep_result.get("count", 0)),
                                        "mean_confidence": round(float(sweep_result.get("mean_confidence", 0.0)), 3),
                                        "runtime_ms": round(runtime_ms, 1),
                                    }
                                )

                        st.session_state.counter_sweep_results = sweep_rows

            if st.session_state.counter_sweep_results:
                st.markdown("### Threshold Sweep Results")
                st.caption("IoU fixed at 0.45 for sweep; choose one confidence and apply.")
                st.dataframe(st.session_state.counter_sweep_results, use_container_width=True)

                conf_options = [float(r["conf"]) for r in st.session_state.counter_sweep_results]
                default_conf = float(st.session_state.get("counter_confidence", confidence))
                default_idx = 0
                for idx, conf_opt in enumerate(conf_options):
                    if abs(conf_opt - default_conf) < 1e-9:
                        default_idx = idx
                        break

                chosen_conf = st.selectbox(
                    "Confidence to apply",
                    options=conf_options,
                    index=default_idx,
                    key="counter_sweep_choice",
                )
                if st.button("Apply Selected Confidence", key="counter_sweep_apply"):
                    st.session_state.counter_confidence = float(chosen_conf)
                    st.success(f"Applied confidence {float(chosen_conf):.2f}.")
                    st.rerun()

            if discard_counter_btn:
                try:
                    Path(st.session_state.counter_preview_path).unlink()
                except Exception:
                    pass
                st.session_state.counter_preview_jpeg = None
                st.session_state.counter_preview_path = None
                st.session_state.counter_quad_points = None
                st.session_state.counter_quad_source = None
                st.session_state.counter_canvas_seed_token = None
                st.session_state.counter_canvas_seed_drawing = None
                st.session_state.counter_last_result = None
                st.session_state.counter_result_logged = False
                st.session_state.counter_sweep_results = []
                st.rerun()

        result = st.session_state.counter_last_result
        if result:
            st.subheader("Count Result")
            count_val = int(result.get("count", 0))
            mean_conf = float(result.get("mean_confidence", 0.0))
            conf_std = float(result.get("confidence_std", 0.0))
            # Heuristic uncertainty: scales with lower confidence and higher variance.
            margin = max(1, int(round(count_val * (0.05 + 0.20 * max(0.0, 1.0 - mean_conf) + 0.10 * conf_std))))

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Estimated Count", count_val)
            with metric_cols[1]:
                st.metric("Estimated Range", f"{max(0, count_val - margin)} - {count_val + margin}")
            with metric_cols[2]:
                st.metric("Mean Confidence", f"{mean_conf:.2f}")
            with metric_cols[3]:
                st.metric("Tiles Processed", int(result.get("tile_count", 0)))

            if result.get("image_with_boxes") is not None:
                st.image(result["image_with_boxes"], caption="Augmented tray count overview", use_container_width=True)

            saved_path = result.get("saved_path")
            if saved_path:
                st.success(f"Annotated image saved: {saved_path}")
                st.caption("Tip: tweaking confidence levels until visually appropriate can give better data.")

            source_path = st.session_state.counter_preview_path
            tray_default = Path(source_path).stem if source_path else "counter_run"
            tray_for_log = st.text_input("Tray ID for log", value=tray_default, key="counter_tray_id_log")
            if st.button("Save Result", type="primary", key="counter_save_result"):
                from datetime import datetime as _dt
                result_settings = result.get("settings", {})
                logged_conf = float(result_settings.get("confidence", confidence))
                logged_iou = float(result_settings.get("iou", iou))
                logged_imgsz = int(result_settings.get("imgsz", tile_size))
                logged_overlap = int(result_settings.get("overlap", overlap))
                logged_mode = str(result_settings.get("mode", "tiled_overlap_global_merge"))
                logged_edge_margin = int(result_settings.get("edge_margin_px", result.get("edge_margin_px", 8)))
                logged_min_box_area = float(result_settings.get("min_box_area_px", result.get("min_box_area_px", 24.0)))
                logged_model = result.get("model_name", model_path.name)

                data_mgr.add_detection(
                    tray_id=tray_for_log,
                    zoom=1,
                    model=logged_model,
                    count=count_val,
                    mean_confidence=mean_conf,
                    image_path=Path(saved_path) if saved_path else (Path(source_path) if source_path else Path("")),
                    notes=(
                        f"margin=\u00b1{margin}; mode={logged_mode}; overlap={logged_overlap}px; "
                        f"edge_margin={logged_edge_margin}px; min_box_area={logged_min_box_area:.0f}"
                    ),
                    batch=st.session_state.counter_batch_number,
                    est_range_low=max(0, count_val - margin),
                    est_range_high=count_val + margin,
                    confidence_thresh=logged_conf,
                    iou_thresh=logged_iou,
                    tile_size=logged_imgsz,
                    overlap=int(logged_overlap),
                )
                st.session_state.counter_batch_data.append({
                    "Batch": st.session_state.counter_batch_number,
                    "Photo #": st.session_state.counter_batch_index,
                    "Tray ID": tray_for_log,
                    "Count": count_val,
                    "Est. Range": f"{max(0, count_val - margin)}\u2013{count_val + margin}",
                    "Mean Conf.": f"{mean_conf:.2f}",
                    "Conf. Thresh": logged_conf,
                    "IoU Thresh": logged_iou,
                    "Tile Size": logged_imgsz,
                    "Overlap px": int(logged_overlap),
                    "Mode": logged_mode,
                    "Edge Margin px": logged_edge_margin,
                    "Min Box Area": f"{logged_min_box_area:.0f}",
                    "Model": logged_model,
                    "Timestamp": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                st.session_state.counter_result_logged = True
                st.success("Result saved to spreadsheet.")
            elif st.session_state.counter_result_logged:
                st.caption("Result already saved for this processed image.")

elif page == "data-tables":
    import pandas as _pd

    st.title("🐛 Data & Tables")
    st.caption("Session batch results, statistics, and the full historical log. Use the download buttons to export to Excel.")

    batch_num = st.session_state.get("counter_batch_number", 1)
    batch_data = st.session_state.get("counter_batch_data", [])

    # ── Current batch ─────────────────────────────────────────────────────────
    st.subheader(f"Batch {batch_num} — Saved Results")

    if not batch_data:
        st.info(
            "No results saved in this batch yet. "
            "Go to **Exuvia Counter**, process images, then click **Save Result**."
        )
    else:
        batch_df = _pd.DataFrame(batch_data)
        st.dataframe(batch_df, use_container_width=True)

        _xlbuf = io.BytesIO()
        batch_df.to_excel(_xlbuf, index=False)
        st.download_button(
            "Download Batch as Excel",
            data=_xlbuf.getvalue(),
            file_name=f"exuvia_batch_{batch_num}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_batch_excel",
        )

        # ── Clear batch ───────────────────────────────────────────────────────
        if st.session_state.get("confirm_clear_batch"):
            st.warning("This will remove all saved results from the current batch. This cannot be undone.")
            cb_col1, cb_col2 = st.columns([1, 4])
            with cb_col1:
                if st.button("Yes, clear batch", type="primary", key="do_clear_batch"):
                    st.session_state["counter_batch_data"] = []
                    st.session_state["counter_batch_number"] = 1
                    st.session_state["confirm_clear_batch"] = False
                    st.rerun()
            with cb_col2:
                if st.button("Cancel", key="cancel_clear_batch"):
                    st.session_state["confirm_clear_batch"] = False
                    st.rerun()
        else:
            if st.button("Clear Batch", key="btn_clear_batch"):
                st.session_state["confirm_clear_batch"] = True
                st.rerun()

        # ── Statistics ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Batch Statistics")

        counts = _pd.to_numeric(batch_df["Count"], errors="coerce").dropna()
        n = len(counts)

        if n > 0:
            mean_c = float(counts.mean())
            std_c  = float(counts.std()) if n > 1 else 0.0
            cv_pct = (std_c / mean_c * 100) if mean_c > 0 else 0.0
            se     = std_c / (n ** 0.5) if n > 1 else 0.0
            ci_low  = mean_c - 1.96 * se
            ci_high = mean_c + 1.96 * se
            moe95 = 1.96 * se if n > 1 else 0.0
            stability = min(100.0, max(0.0, 100.0 * (1.0 - (1.0 / (n ** 0.5)))))

            sc = st.columns(4)
            with sc[0]:
                st.metric("Photos Saved", n)
                st.metric("Total Counted", int(counts.sum()))
            with sc[1]:
                st.metric("Mean Count", f"{mean_c:.1f}")
                st.metric("Std Deviation", f"{std_c:.1f}")
            with sc[2]:
                st.metric("Min", int(counts.min()))
                st.metric("Max", int(counts.max()))
            with sc[3]:
                st.metric(
                    "Coeff. of Variation", f"{cv_pct:.1f}%",
                    help="Std dev as % of mean — lower = more consistent counts across photos.",
                )
                if n > 1:
                    st.metric(
                        "95% CI of Mean",
                        f"{max(0.0, ci_low):.0f} – {ci_high:.0f}",
                        help="Range where the true mean count likely falls (based on this batch).",
                    )

            if n > 1:
                clt_col1, clt_col2 = st.columns(2)
                with clt_col1:
                    st.metric(
                        "95% Margin of Error",
                        f"±{moe95:.1f}",
                        help="As sample size grows, this shrinks roughly with 1/sqrt(n).",
                    )
                with clt_col2:
                    st.metric(
                        "Batch Stability",
                        f"{stability:.0f}%",
                        help="Heuristic indicator that increases with larger sample size.",
                    )

            st.caption(
                "**CV** (Coefficient of Variation): under 15% = very consistent, under 30% = acceptable.  "
                "**95% CI**: if you repeated this batch many times, the mean would fall in this range 95% of the time."
            )

            st.markdown("---")
            st.subheader("Quick Visuals")
            viz_df = batch_df.copy()
            viz_df["Count"] = _pd.to_numeric(viz_df["Count"], errors="coerce")
            viz_df = viz_df.dropna(subset=["Count"]).reset_index(drop=True)
            if not viz_df.empty:
                viz_df["Sample #"] = viz_df.index + 1
                viz_df["Rolling Mean (n=5)"] = viz_df["Count"].rolling(window=5, min_periods=1).mean()
                st.caption("Run chart of counts and rolling average.")
                st.line_chart(viz_df.set_index("Sample #")[["Count", "Rolling Mean (n=5)"]], height=260)

                st.caption("Count frequency per sample (simple histogram-style bar chart).")
                freq = viz_df["Count"].value_counts().sort_index()
                st.bar_chart(freq, height=240)

                if n > 1 and std_c > 0:
                    import numpy as _np
                    st.caption("CLT view: expected 95% margin of error vs number of photos (falls as n grows).")
                    n_grid = _np.arange(2, max(21, n + 1))
                    moe_vals = 1.96 * std_c / _np.sqrt(n_grid)
                    moe_df = _pd.DataFrame({"Sample Size (n)": n_grid, "95% Margin of Error": moe_vals})
                    st.line_chart(moe_df.set_index("Sample Size (n)"), height=220)

        # ── Count distribution chart ──────────────────────────────────────────
        if n >= 1:
            st.markdown("---")
            st.subheader("Count Distribution")
            st.caption(
                "Each bar shows how many photos had a count in that range. "
                "The orange curve is a fitted normal distribution — "
                "if the bars follow the curve, your counts are nicely consistent."
            )
            try:
                import numpy as _np
                bins = max(3, min(12, int(_np.sqrt(n)) + 1))
                hist_counts, hist_edges = _np.histogram(counts, bins=bins)
                hist_labels = [f"{hist_edges[i]:.0f}-{hist_edges[i+1]:.0f}" for i in range(len(hist_counts))]
                hist_df = _pd.DataFrame({"Count Range": hist_labels, "Photos": hist_counts}).set_index("Count Range")
                st.bar_chart(hist_df, height=240)
            except Exception:
                pass

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                import numpy as _np
                from scipy.stats import norm as _norm, shapiro as _shapiro

                fig, ax = _plt.subplots(figsize=(8, 4))
                n_bins = max(3, min(max(1, int(n / 2)), 12))
                ax.hist(
                    counts, bins=n_bins, density=True,
                    color="#4c8cbf", alpha=0.72, edgecolor="white",
                    label="Observed counts",
                )

                mu    = counts.mean()
                sigma = counts.std() if n > 1 else 0.0
                if sigma > 0:
                    x_vals = _np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
                    ax.plot(
                        x_vals, _norm.pdf(x_vals, mu, sigma),
                        color="#e05a00", linewidth=2.2,
                        label=f"Normal fit   μ = {mu:.1f},  σ = {sigma:.1f}",
                    )
                    ax.axvline(mu, color="#e05a00", linestyle="--", linewidth=1.2, alpha=0.85)
                    ax.axvline(mu - sigma, color="#666", linestyle=":", linewidth=1.1, alpha=0.7,
                               label="μ ± 1σ  (≈68% of data)")
                    ax.axvline(mu + sigma, color="#666", linestyle=":", linewidth=1.1, alpha=0.7)
                    ax.axvspan(mu - sigma, mu + sigma, alpha=0.07, color="#e05a00")

                ax.set_xlabel("Exuvia Count per Photo", fontsize=11)
                ax.set_ylabel("Probability Density", fontsize=11)
                ax.set_title(f"Batch {batch_num} — Count Distribution", fontsize=12, fontweight="bold")
                ax.legend(fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                _plt.close(fig)

                # Shapiro-Wilk normality check — only meaningful at n ≥ 8
                if n >= 8 and sigma > 0:
                    _, p_val = _shapiro(counts)
                    if p_val > 0.05:
                        st.success(
                            f"Shapiro-Wilk test: p = {p_val:.3f} (> 0.05) — "
                            "counts are consistent with a normal distribution. ✔"
                        )
                    else:
                        st.warning(
                            f"Shapiro-Wilk test: p = {p_val:.3f} (≤ 0.05) — "
                            "counts may not be normally distributed. "
                            "Could indicate real tray-to-tray variation or a few outlier images."
                        )

                if n > 1 and sigma > 0:
                    st.caption(
                        f"About 68% of counts are expected between "
                        f"**{max(0, mu - sigma):.0f}** and **{mu + sigma:.0f}**, "
                        f"and 95% between **{max(0, mu - 2*sigma):.0f}** and **{mu + 2*sigma:.0f}**."
                    )
            except ImportError as _ie:
                st.warning(f"Chart unavailable (missing matplotlib/scipy): {_ie}")

    # ── Full history log ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full History Log")
    history_df = data_mgr.get_dataframe()
    if history_df.empty:
        st.info("No historical records logged yet.")
    else:
        st.dataframe(history_df, use_container_width=True)

        _hbuf = io.BytesIO()
        history_df.to_excel(_hbuf, index=False)
        st.download_button(
            "Download Full Log as Excel",
            data=_hbuf.getvalue(),
            file_name="exuvia_full_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_full_log",
        )

        # ── Clear full log ────────────────────────────────────────────────────
        if st.session_state.get("confirm_clear_log"):
            st.warning("This will permanently delete all historical records from the log file. This cannot be undone.")
            cl_col1, cl_col2 = st.columns([1, 4])
            with cl_col1:
                if st.button("Yes, clear log", type="primary", key="do_clear_log"):
                    data_mgr.clear_all()
                    st.session_state["confirm_clear_log"] = False
                    st.rerun()
            with cl_col2:
                if st.button("Cancel", key="cancel_clear_log"):
                    st.session_state["confirm_clear_log"] = False
                    st.rerun()
        else:
            if st.button("Clear Full Log", key="btn_clear_log"):
                st.session_state["confirm_clear_log"] = True
                st.rerun()

elif page == "technical-info":
    st.title("🐛 Info")
    st.markdown(
        "[Guide/Manual](https://docs.google.com/document/d/1wDBeD1E9GZ5ZaHm4I_H8D6K9ydvff_BBw3AuetpO-Ec/edit?usp=sharing) "
        "— Full documentation and operational guide."
    )
    st.markdown("---")
    st.markdown("""
Architecture:
- Hardware: Raspberry Pi (Pi 5), Raspberry Pi HQ Camera, optional remote client over LAN/Tailscale.
- App layer: Streamlit single-page workflow with sidebar navigation.
- Camera stack: `rpicam-jpeg` for full-resolution stills and `rpicam-still` timelapse preview stream.
- CV stack: Ultralytics YOLO model from the local `models/` folder with tiled inference and merge/dedup.
- Storage: Local filesystem (`data/captures`, `data/tiles`, `data/detections`, `data/counter_inputs`).
- Logging: Excel-based records via pandas (`data/exuvia_log.xlsx`).

Operational procedure:
1. Use Video Capture (Focus) to adjust lens manually.
2. Capture tray photos in Image Capture or Exuvia Counter.
3. Run Exuvia Counter processing (tile, detect, merge, count).
4. Review annotation quality and tune confidence/IoU if needed.
5. Save results to the spreadsheet and review batch statistics in Data & Tables.

Key technologies:
- Python, Streamlit, OpenCV, NumPy, pandas, SciPy, Matplotlib.
- Ultralytics YOLO11 model family for detection.
- systemd headless service for boot-time startup.

Design constraints:
- Camera access is single-process; only one active camera owner is supported.
- Inference on Pi can take minutes for dense trays; processing is asynchronous in UI.
- Tiled detection improves small-object recall but requires duplicate-box suppression.
""")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>Exuvia Counting System - UBCO 25/26</small>
</div>
""", unsafe_allow_html=True)
