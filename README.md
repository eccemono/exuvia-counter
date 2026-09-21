# OKSIR Exuvia Counter

Automating the counting of codling moth exuvia for British Columbia's sterile insect release program. A Raspberry Pi imaging system, a YOLO computer vision model, and a Streamlit web app that turns a slow manual task into a minute of camera work.

## The problem

The Okanagan Kootenay Sterile Insect Release program (OKSIR) breeds and releases millions of sterile codling moths to protect Okanagan orchards without relying on pesticides. To verify production, staff count *exuvia*, the shed exoskeletons larvae leave behind when they mature into adult moths.

Counting is done by hand. Staff divide each tray into quarters, count one quarter, and multiply by four to estimate the total. It is slow, tedious, and prone to error, and it can tie up a full employee for a summer.

This project replaces the manual method with a controlled imaging box and a computer vision model that estimates a tray count in minutes.

## What it is

- An illuminated imaging enclosure built around a Raspberry Pi 5 and a 12.3 MP HQ camera
- A YOLO detection model trained on over 15,000 labeled exuvia
- A Streamlit web app for capture, counting, and data export
- Tiled inference with overlap and duplicate suppression for small object recall

## Key numbers

| Metric | Value |
| ------ | ----- |
| Detection accuracy (mAP@50) | 92% |
| Precision and recall | above 90% |
| Labeled training samples | 15,000+ |
| Target | outperform the manual quarter count on both speed and accuracy |

## Hardware

- Raspberry Pi 5, headless on boot via a systemd service
- Raspberry Pi HQ Camera (12.3 MP Sony IMX477)
- 12 MP factory automation lens on a C mount
- White and UV LED array inside an aluminum and vinyl coated enclosure
- 3D printed overhead camera mount with adjustable focus, iris, and zoom

## Software

- Python and Streamlit for the web interface
- Ultralytics YOLO for detection
- OpenCV, NumPy, pandas, SciPy, and Matplotlib for processing and statistics
- Full resolution stills via `rpicam-jpeg` and a live preview via `rpicam-still`

## Features

- Live camera preview for manual focus
- Full resolution image capture
- Exuvia counter with adjustable confidence and IoU thresholds
- Threshold sweep to tune detection against a real tray
- Batch tracking with statistics
- One click export to Excel

## Project structure

```
exuvia-counter/
├── app.py             # Main Streamlit application
├── camera.py          # Camera control (Pi HQ camera and USB fallback)
├── detector.py        # YOLO detection and tiled inference
├── tiler.py           # Image tiling for training data
├── data_manager.py    # Excel logging and statistics
├── config.py          # Default settings
├── requirements.txt   # Core dependencies
├── requirements-ml.txt# Optional YOLO dependencies
├── run.sh             # Setup and launch script
├── examples.py        # Programmatic usage examples
└── .streamlit/        # Streamlit theme configuration
```

## Getting started

### Prerequisites

- Raspberry Pi 4 or 5 (or a desktop for testing)
- Python 3.11 or 3.12 recommended for YOLO
- A Raspberry Pi HQ Camera or a USB webcam

### Install and run

```bash
# Install dependencies
pip install -r requirements.txt

# Optional YOLO support
pip install -r requirements-ml.txt

# Run the app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or use the launcher, which creates a virtual environment and installs dependencies for you:

```bash
./run.sh
```

Then open `http://localhost:8501`. The app is reachable from any device on the same network, and optionally over Tailscale.

## Models

The trained YOLO weights are not included in this repository. Place `.pt` files in a `models/` directory next to the app and select them from the counter page. The model loader picks up any `.pt` file in that folder.

## Usage

1. Open Video Capture to focus the lens against a real tray.
2. Capture tray photos in Image Capture or Exuvia Counter.
3. Run the counter to tile, detect, merge, and count.
4. Review the bounding boxes and tune confidence or IoU if needed.
5. Save results and review batch statistics in Data & Tables.

## How detection works

The model runs tiled inference over the full resolution image with overlap between tiles. Detections are remapped to full image coordinates, then passed through non-maximum suppression and center based deduplication to remove duplicate boxes from adjacent tiles. Edge hugging and very small boxes are filtered before the final count.

## Notes and disclaimer

Built as an engineering capstone project for OKSIR. The code is provided as is for reference and long term maintenance. It assumes a Raspberry Pi camera stack and was validated on site at the client facility.

## Technology

Python, Streamlit, OpenCV, NumPy, pandas, SciPy, Matplotlib, Ultralytics YOLO, and the Raspberry Pi camera tooling.
