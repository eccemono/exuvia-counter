# 📚 Exuvia Counter - Complete Index & Quick Reference

## 🚀 START HERE

**First time?** Read in this order:
1. **QUICKSTART.md** ← 5 minutes to get running
2. **PROJECT_SUMMARY.md** ← Understand what you built
3. **README.md** ← Full detailed docs

**Just want to run it?**
```bash
cd exuvia_app
bash run.sh
```
Then open: `http://localhost:8501`

---

## 📁 File Guide

### Core Application
| File | Purpose | Read When |
|------|---------|-----------|
| **app.py** | Main Streamlit app (THE PROGRAM) | Want to understand the UI |
| **camera.py** | Camera control & capture | Need camera help |
| **detector.py** | YOLO detection engine | Want to change model |
| **tiler.py** | Image → tiles converter | Creating training data |
| **data_manager.py** | Excel logging & stats | Need data analytics |
| **config.py** | Configuration settings | Want to customize without coding |

### Documentation
| File | Purpose | Best For |
|------|---------|----------|
| **QUICKSTART.md** | 5-minute setup | Getting started fast |
| **README.md** | Complete reference | Understanding all features |
| **PROJECT_SUMMARY.md** | High-level overview | Seeing the big picture |
| **examples.py** | Code samples | Learning programmatic use |
| **requirements.txt** | Python packages | Understanding dependencies |

### Configuration
| File | Purpose |
|------|---------|
| **.streamlit/config.toml** | Streamlit UI settings |
| **run.sh** | Quick start script |

### Data (Auto-created)
```
data/
├── captures/          ← Your camera photos
├── tiles/             ← Training data tiles
├── detections/        ← Detection results
├── exports/           ← Excel file exports
└── exuvia_log.xlsx    ← Main database
```

---

## 💡 Common Tasks

### "I want to capture images"
→ Go to **Live Capture** page in app.py

### "I want to set up training data"
→ Go to **Training Data** page, see README.md Section on Tiling

### "I want to see data analytics"
→ Go to **Data Analysis** page

### "I want to change the YOLO model"
→ Edit **config.py** line: `DEFAULT_MODEL = "yolov8s.pt"`

### "I want to run detection on my own images"
→ Use **examples.py** → Example 6 (detect_local_image)

### "I want to understand how detection works"
→ Read **detector.py** docstrings + README.md

### "I want to process multiple images automatically"
→ Use **examples.py** → Example 2 (batch_process)

### "I want to export data to Excel"
→ Data Analysis page (built-in) OR **examples.py** → Example 5

### "I need to adjust confidence threshold"
→ Live Capture page has slider (dynamic)

### "I want to debug something"
→ Check terminal output where you ran `streamlit run app.py`

---

## 🎯 Three-Stage Workflow

### Stage 1: Test & Calibrate (Now)
```
Run app → Live Capture page → Take photos → Adjust zoom/confidence
```
Files used: camera.py, detector.py, app.py

### Stage 2: Build Training Data (1-2 weeks)
```
Capture high-res images → Training Data page → Create tiles
→ Label on Roboflow → Train custom YOLO
```
Files used: camera.py, tiler.py, detector.py

### Stage 3: Deploy (Ready to go)
```
Use custom model → Run detections → Track in Excel
→ Export for Scott's analysis
```
Files used: all

---

## 🔧 Customization Quick Ref

### Change Camera
**File:** camera.py
```python
cam = get_camera(use_pi_camera=False)  # Use USB instead of Pi
```

### Change Model Size
**File:** config.py
```python
DEFAULT_MODEL = "yolov8m.pt"  # nano (n), small (s), medium (m)
```

### Change Tile Size
**File:** config.py
```python
DEFAULT_TILE_SIZE = 512  # Larger tiles (256-512 px)
DEFAULT_OVERLAP = 100    # More overlap
```

### Change Detection Confidence
**In app:** Live Capture page → slider (0.0-1.0)

### Change Data Location
**File:** config.py
```python
LOG_FILE = Path("my_folder/data.xlsx")
```

---

## 📊 Excel Data Format

Your `data/exuvia_log.xlsx` columns:

```
Timestamp | Tray_ID | Zoom | Model | Count | Mean_Confidence | Image_Path | Notes
```

Use in Excel:
- Pivot tables
- Charts
- Filters
- Manual edits

Use in Python:
```python
import pandas as pd
df = pd.read_excel("data/exuvia_log.xlsx")
```

---

## 🐛 Common Issues & Fixes

| Problem | Fix | Details |
|---------|-----|---------|
| Camera not detected | Check cable, try `libcamera-still` | camera.py line ~50 |
| YOLO model won't download | Check internet, disk space | detector.py line ~30 |
| Can't access from Windows PC | Check network, get Pi IP `hostname -I` | README.md |
| Tiles won't create | Check image file exists | tiler.py line ~20 |
| Data won't save | Check folder permissions, disk space | data_manager.py |
| Streamlit won't run | `pip install -r requirements.txt` | requirements.txt |

---

## 📞 Help Resources

| Question | Where to Look |
|----------|---------------|
| How do I run the app? | QUICKSTART.md |
| What does this code do? | README.md + docstrings in each file |
| How do I customize it? | config.py + PROJECT_SUMMARY.md |
| What are the advantages? | PROJECT_SUMMARY.md |
| Can I use it without Streamlit? | examples.py |
| How does detection work? | detector.py + ultralytics.com |
| How do I improve accuracy? | README.md → Phase 2 section |

---

## 🎓 Learning Path

**Beginner (just use it):**
1. Run `bash run.sh`
2. Go through each page
3. Capture some images
4. Look at Excel data

**Intermediate (customize):**
1. Read PROJECT_SUMMARY.md
2. Edit config.py
3. Understand each module (docstrings)
4. Run examples.py

**Advanced (extend it):**
1. Read all code
2. Understand Streamlit architecture
3. Add features to app.py
4. Create your own detection pipeline

---

## ✅ Checklist Before Deploying to Osoyoos

- [ ] Test capture on Pi with HQ camera
- [ ] Collect 100+ sample images
- [ ] Verify detection accuracy
- [ ] Create training tiles
- [ ] Label tiles and create custom YOLO model
- [ ] Test custom model detection
- [ ] Test Excel export
- [ ] Test network access from Scott's PC
- [ ] Document any custom settings in config.py
- [ ] Backup all code to git (if using version control)

---

## 📦 What You Have

✅ **One unified program** (app.py)  
✅ **Web interface** (no setup on Windows PC)  
✅ **3 detection models** (nano/small/medium)  
✅ **Zoom levels** (1x/2x/3x)  
✅ **Training data pipeline** (automatic tiling)  
✅ **Excel logging** (comprehensive data tracking)  
✅ **Statistical analysis** (outliers, distributions)  
✅ **Production-ready code** (can deploy now)  

---

## 🎯 Next Steps

**Do this first:**
1. Run the app: `bash run.sh`
2. Read QUICKSTART.md
3. Take 5 test photos
4. Check data/exuvia_log.xlsx

**Then do this:**
5. Collect 100+ real images from trays
6. Create training tiles
7. Label tiles on Roboflow
8. Train custom YOLO model
9. Replace DEFAULT_MODEL in config.py

**Finally:**
10. Deploy to Osoyoos facility
11. Run regular captures
12. Export data for Scott

---

**Questions? Check the relevant .md file above first!**

---

Made with ❤️ for Osoyoos facility 🐛
