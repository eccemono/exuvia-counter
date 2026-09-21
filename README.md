# 🐛 Exuvia Counter - Complete System

A unified Python application for exuvia (insect shell) counting using Raspberry Pi, HQ Camera, and YOLO object detection. Includes live preview, training data management, and comprehensive data analytics with Excel export.

## Project Structure

```
exuvia_app/
├── app.py                  # Main Streamlit application (THE PROGRAM)
├── camera.py               # Picamera2 wrapper for Pi HQ camera
├── detector.py             # YOLO v8 detection engine
├── tiler.py                # Image tiling for training data
├── data_manager.py         # Excel logging & statistics
├── requirements.txt        # Python dependencies
├── data/                   # Auto-created directories
│   ├── captures/           # Full resolution images
│   ├── tiles/              # Training tiles by tray
│   ├── detections/         # Annotated detection results
│   ├── exports/            # Exported Excel files
│   └── exuvia_log.xlsx     # Main data log
└── README.md
```

## Features

### 📷 Live Capture Page
- **Real-time camera preview** from Raspberry Pi HQ Camera
- **1x, 2x, 3x zoom levels** (digital zoom)
- **YOLO detection** with adjustable confidence threshold
- **Model selection**: Nano (fast), Small (balanced), Medium (accurate)
- **Automatic data logging** to Excel
- **Keyboard shortcuts** for quick capture (spacebar support in future versions)

### 🏋️ Training Data Page
- **Image tiling**: Convert full images into 256px tiles with overlap
- **Tile organization**: Automatically organized by tray ID
- **Tile browser**: View all tiles in your library
- **Streamlined for labeling**: Export tiles ready for Roboflow, Label Studio, or similar

### 📊 Data Analysis Page
- **Real-time statistics**: Total captures, counts, averages, std dev
- **Tray breakdown**: Per-tray summaries
- **Visualizations**: Time series and distribution charts
- **Outlier detection**: Automatically find anomalies (Z-score analysis)
- **Excel export**: Download data in multiple formats
- **Statistical analysis**: Ready for further statistical testing

## Installation & Setup

### Prerequisites
- Raspberry Pi 4/5 (or desktop for testing)
- Picamera2 library (Pi only)
- Python 3.8+

Recommended runtime:
- Python 3.11/3.12 for best YOLO compatibility on Raspberry Pi.
- Python 3.13 works for capture/logging/analytics, but YOLO wheels may be unavailable.

### 1. Install on Raspberry Pi

```bash
# SSH into your Pi, then:
cd ~/Desktop/Exuvia/Exuvia-Counting-main\ \(1\)/Exuvia-Counting-main/exuvia_app

# Install dependencies
pip install -r requirements.txt

# Optional YOLO dependencies
pip install -r requirements-ml.txt

# Run the app
streamlit run app.py
```

The app will start a local server. Note the URL (usually: `http://localhost:8501`)

### 2. Access from Windows PC

If Pi is on network at Osoyoos facility:

1. Find Pi's IP address:
   ```bash
   # On Pi terminal
   hostname -I
   ```
   
   This gives you something like: `192.168.1.100`

2. On Windows PC, open browser and go to:
   ```
   http://192.168.1.100:8501
   ```

3. That's it! No installation needed on Windows. Just a browser.

### 2.1 Access from Anywhere with Tailscale

If both devices are in your tailnet, use one of these URLs:

- `http://<tailscale-ip>:8501`
- `http://<pi-tailnet-name>:8501`

The app now shows these links in the sidebar under **Links**, and `run.sh` prints them on startup.

If you want a public HTTPS link (without installing Tailscale on the viewer device), you can use Funnel:

```bash
./enable_funnel.sh
tailscale funnel status
```

That returns an HTTPS URL you can open from anywhere.

### 3. Testing on Desktop (Without Pi Camera)

The system gracefully falls back to USB webcam if Picamera2 is unavailable:

```bash
# Will work with any USB webcam
streamlit run app.py
```

## How to Use

### Workflow 1: Capture & Detect

1. **Go to "Live Capture" page**
2. **Select zoom level** (1x for full view, 2x/3x for focusing)
3. **Enter tray ID** (e.g., "tray_001")
4. **Add optional notes**
5. **Click "Capture Image"** - frame is saved + detection runs
6. **Results appear instantly** - count, confidence, annotated image
7. **Data auto-logs to Excel**

### Workflow 2: Prepare Training Data

1. **Go to "Training Data" page**
2. **Upload a high-res image** from your tray
3. **Set tray ID** + tile size (default 256px is good)
4. **Click "Create Tiles"** - image is split into overlapping tiles
5. **Tiles saved to** `data/tiles/{tray_id}/`
6. **Later:** Export tiles and use Roboflow/Label Studio to label them
7. **Once labeled:** Re-train YOLO with your custom data

### Workflow 3: Analyze Data

1. **Go to "Data Analysis" page**
2. **Summary stats** show at top (total captures, avg count, etc.)
3. **Tray breakdown** shows per-tray statistics
4. **Charts** visualize trends over time
5. **Outlier detection** finds unusual readings
6. **Export to Excel** for Scott to use in reports/spreadsheets

## File Descriptions

### `app.py` (Main Program)
- Streamlit web application
- Handles UI, navigation between 3 pages
- Integrates all modules
- **This is what you run:** `streamlit run app.py`

### `camera.py`
- `CameraManager` class: Unified interface for Pi or USB camera
- `get_frame()`: Returns current frame
- `capture_image()`: Saves image to disk
- Falls back gracefully if Picamera2 unavailable

### `detector.py`
- `ExuviaDetector` class: YOLO inference wrapper
- `detect()`: Runs model on image, returns count + bboxes
- Supports YOLOv8n, YOLOv8s, YOLOv8m models
- Auto-downloads model on first use (~20-100MB depending on size)

### `tiler.py`
- `ImageTiler` class: Split images into tiles
- `create_tiles()`: Generates overlapping tiles
- `save_tiles()`: Stores tiles organized by tray
- `load_tile_library()`: Browse all tiles

### `data_manager.py`
- `DataManager` class: Excel logging + statistics
- `add_detection()`: Log a detection record
- `get_summary_stats()`: Calculate summary metrics
- `detect_outliers()`: Find anomalies (Z-score)
- Auto-saves to `data/exuvia_log.xlsx`

## Configuration

### Change Model Size

In app.py, "Data Analysis" page:
```python
model_choice = st.selectbox(
    "Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],  # Change here
)
```

- **yolov8n** (nano): ~6MB, fastest, ok accuracy
- **yolov8s** (small): ~22MB, balanced
- **yolov8m** (medium): ~50MB, most accurate but slower

### Change Tile Size

In "Training Data" page, adjust:
- **Tile Size**: 128-512px (256 recommended)
- **Overlap**: 0-100px (50 recommended to avoid cutting exuvia)

### Change Confidence Threshold

In "Live Capture" page:
- **Lower (0.3)**: More detections, some false positives
- **Default (0.5)**: Good balance
- **Higher (0.8)**: Only very confident detections

## Excel Data Format

Your `data/exuvia_log.xlsx` will have columns:

| Timestamp | Tray_ID | Zoom | Model | Count | Mean_Confidence | Image_Path | Notes |
|-----------|---------|------|-------|-------|-----------------|------------|-------|
| 2026-03-16 14:30:45 | tray_001 | 1x | YOLOv8N | 12 | 0.87 | data/captures/tray_001_... | Checked manually |
| 2026-03-16 14:31:20 | tray_001 | 2x | YOLOv8N | 11 | 0.89 | data/captures/tray_001_... | - |

You can pivot/analyze this in Excel or pandas:
```python
import pandas as pd
df = pd.read_excel("data/exuvia_log.xlsx")
print(df.groupby("Tray_ID")["Count"].mean())  # Average per tray
print(df[df["Count"] > df["Count"].std() * 2])  # Outliers
```

## Troubleshooting

### Camera not detected
```
❌ Camera not available. Check connection and restart app.
```
**Fix:** 
- Check SSH connection to Pi
- Verify camera cable is seated
- Test: `libcamera-still --help` on Pi terminal

### YOLO model won't download
```
ultralytics.utils.errors.HUBModelError: ...
```
**Fix:**
- Model downloads on first use (~100MB)
- Ensure internet connection
- Check disk space: `df -h`

### Streamlit not found
```
command not found: streamlit
```
**Fix:**
```bash
pip install -r requirements.txt
```

### Picamera2 not available
```
WARNING: Picamera2 not available. Fallback to OpenCV webcam.
```
**This is OK!** System falls back to USB webcam. Only an issue on Pi if you want to use HQ camera.

## Next Steps

### Phase 2: Custom Model Training
1. Collect 100+ tray images
2. Use tiling to create 1000+ tiles
3. Label tiles in Roboflow (~2-3 hours)
4. Train YOLOv8 on custom data (1-2 hours on Pi or PC)
5. Replace model path in `detector.py`

### Phase 3: Advanced Features
- Web UI accessible from anywhere (deploy to cloud)
- Model versioning & A/B testing
- Automated alerts if counts are abnormal
- Integration with Scott's reporting system
- Import/export from Google Sheets

## Technology Stack

- **Streamlit**: Web framework (no frontend coding needed)
- **OpenCV**: Image processing
- **Ultralytics YOLOv8**: Object detection
- **Pandas**: Data management
- **Scipy**: Statistical analysis
- **Picamera2**: Raspberry Pi camera interface

## Questions?

Refer to:
- **Streamlit docs**: https://docs.streamlit.io
- **YOLOv8 docs**: https://docs.ultralytics.com
- **Picamera2 docs**: https://github.com/raspberrypi/picamera2
- **Pandas docs**: https://pandas.pydata.org

---

**Made for Osoyoos facility exuvia counting** 🐛
