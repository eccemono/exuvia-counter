"""
Configuration file for Exuvia Counter
Modify settings here instead of editing code
"""
import json
from pathlib import Path

# ===== CAMERA SETTINGS =====
USE_PI_CAMERA = True  # Set to False to use USB webcam
DEFAULT_ZOOM = 1  # Start at 1x (options: 1, 2, 3)

# ===== DETECTION SETTINGS =====
DEFAULT_MODEL = "yolov8n.pt"  # Options: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"
DEFAULT_CONFIDENCE = 0.5  # Range: 0.0 to 1.0
DEFAULT_IOU = 0.45  # IoU threshold for NMS

# ===== TILING SETTINGS =====
DEFAULT_TILE_SIZE = 256  # Size in pixels (128-512)
DEFAULT_OVERLAP = 50  # Overlap in pixels (0-100)

# ===== DATA STORAGE =====
DATA_DIR = Path("data")
CAPTURES_DIR = DATA_DIR / "captures"
TILES_DIR = DATA_DIR / "tiles"
DETECTIONS_DIR = DATA_DIR / "detections"
EXPORTS_DIR = DATA_DIR / "exports"
LOG_FILE = DATA_DIR / "exuvia_log.xlsx"

# ===== DISPLAY SETTINGS =====
PREVIEW_WIDTH = 1920
PREVIEW_HEIGHT = 1440
THEME_COLOR = "#1f77b4"

# ===== DEFAULTS =====
DEFAULT_TRAY_ID = "tray_001"
MAX_IMAGE_RETRIES = 3

# ===== EXPORT OPTIONS =====
EXCEL_COLUMNS = [
    "Timestamp",
    "Tray_ID",
    "Zoom",
    "Model",
    "Count",
    "Mean_Confidence",
    "Image_Path",
    "Notes"
]

# ===== STATISTICS =====
OUTLIER_ZSCORE_THRESHOLD = 2.0  # Standard deviation multiplier
MIN_DATA_POINTS_FOR_STATS = 2


def get_config():
    """Get configuration as dictionary"""
    return {
        "camera": {
            "use_pi": USE_PI_CAMERA,
            "zoom": DEFAULT_ZOOM
        },
        "detection": {
            "model": DEFAULT_MODEL,
            "confidence": DEFAULT_CONFIDENCE,
            "iou": DEFAULT_IOU
        },
        "tiling": {
            "tile_size": DEFAULT_TILE_SIZE,
            "overlap": DEFAULT_OVERLAP
        },
        "paths": {
            "data": str(DATA_DIR),
            "captures": str(CAPTURES_DIR),
            "tiles": str(TILES_DIR),
            "detections": str(DETECTIONS_DIR),
            "exports": str(EXPORTS_DIR),
            "log": str(LOG_FILE)
        }
    }


def save_user_config(settings_dict, filename="user_settings.json"):
    """Save user customizations to JSON file"""
    with open(filename, "w") as f:
        json.dump(settings_dict, f, indent=2)


def load_user_config(filename="user_settings.json"):
    """Load user customizations from JSON file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
