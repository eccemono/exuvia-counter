# Local Setup Guide

## Quick Start

1. **Clone the repository** (already done):
   ```bash
   cd ~/exuvia-counter
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO models** (get from Pi or train new ones):
   - Place `.pt` files in `models/` directory
   - Or use the provided models from the Pi

5. **Run the app**:
   ```bash
   ./run.sh
   ```
   Or manually:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

6. **Access the app**:
   - Local: http://localhost:8501
   - Network: http://<your-ip>:8501

## Project Structure

- `app.py` - Main Streamlit application (76KB)
- `camera.py` - Camera module (Picamera2/USB webcam)
- `detector.py` - YOLO-based exuvia detection
- `data_manager.py` - Data management and export
- `tiler.py` - Image tiling for large images
- `config.py` - Configuration settings
- `requirements.txt` - Python dependencies
- `requirements-ml.txt` - Optional ML dependencies

## Notes

- Models are not included in the repo (too large)
- Data directory is gitignored (contains detection results)
- Virtual environment is gitignored

## Troubleshooting

If camera doesn't work:
- Check if Picamera2 is installed: `pip install picamera2`
- Or use USB webcam: modify `camera.py` to use `use_pi_camera=False`

If YOLO models fail to load:
- Install ultralytics: `pip install ultralytics`
- Download models or train new ones
