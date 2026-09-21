# Exuvia Counter - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Open Terminal

On **Raspberry Pi** (or your desktop for testing):

```bash
cd ~/Desktop/Exuvia/Exuvia-Counting-main\ \(1\)/Exuvia-Counting-main/exuvia_app
```

### Step 2: Run Setup

**Option A: Using the script (easiest)**
```bash
./run.sh
```

If you also want to attempt YOLO setup in the same command:
```bash
./run.sh --with-ml
```

**Option B: Manual setup**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Note on Python 3.13:
- Base app (capture + logging + analytics) works with `requirements.txt`.
- YOLO may require additional wheels that are often unavailable on Pi + 3.13.
- Optional YOLO install attempt:
```bash
pip install -r requirements-ml.txt
```

### Step 3: Open Browser

Your terminal will show:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XXX:8501
```

If Tailscale is installed and connected on the Pi, the startup script also prints:
```
Tailscale access: http://100.x.y.z:8501
```
You can open that from any device connected to your tailnet.

**On same machine:**
- Go to: `http://localhost:8501`

**From Windows PC (if on same network):**
- Go to: `http://192.168.1.XXX:8501`
  - Replace `192.168.1.XXX` with your Pi's actual IP

**From anywhere (via Tailscale):**
- Open the `Tailscale access` URL shown at startup
- Or use your Pi's Tailscale DNS name with port `8501`

**Public HTTPS (no Tailscale client required):**
```bash
./enable_funnel.sh
```
Then open the HTTPS URL shown by `tailscale funnel status`.

If you want one command that starts everything and enables public access:
```bash
./run_public.sh
```
This starts the app in background (if needed), enables Funnel, and prints the public HTTPS URL.

That's it! You should see the app.

---

## 📋 First-Time Checklist

- [ ] Camera is connected to Pi
- [ ] Python 3.8+ is installed
- [ ] You're in the `exuvia_app` directory
- [ ] Internet connection (for downloading YOLO model on first run)
- [ ] At least 100MB free disk space

---

## 🎮 What to Do First

### Test capture (no model trained yet):
1. Go to **"Live Capture"** page
2. Click **"Capture Image"** button
3. Select an image in `data/captures/`
4. Set a tray ID
5. Hit capture

### Create training tiles:
1. Go to **"Training Data"** page
2. Upload a sample image
3. Click **"Create Tiles"**

### View your data:
1. Go to **"Data Analysis"** page
2. All your captures appear here
3. Automatic stats + charts

---

## 🛠️ Troubleshooting

### "command not found: pip"
```bash
python3 -m pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit==1.28.0
```

### Camera shows "No camera available"
On Raspberry Pi, test:
```bash
libcamera-still --help
```

If this fails, your camera isn't detected. Check:
- Cable connected firmly
- Enable camera in `raspi-config`
- Restart Pi

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port=8502
```

### App is slow on first run
The YOLO model downloads (~20-100MB). First run takes longer.

---

## 📁 File Organization

Once you run the app, it creates:
```
data/
├── captures/            ← Your full images
├── tiles/               ← Training tiles
├── detections/          ← Results with bboxes
├── exports/             ← Excel exports
└── exuvia_log.xlsx      ← Main database
```

All auto-created. Don't delete `exuvia_log.xlsx` unless you want to start over!

---

## 💡 Tips

- **Use 2x/3x zoom** to focus the camera before capturing
- **Adjust confidence** if you get too many/too few detections
- **Export data regularly** - it's backed up to Excel automatically
- **Create tiles from high-res images** for best training data

---

## Next Steps

Once the app is working:

1. **Capture 10+ images** from different trays
2. **Review detection accuracy** - adjust confidence if needed
3. **Collect training data** - create tiles from your best images
4. **Label tiles** on Roboflow (roboflow.com) - free tier available
5. **Train custom model** with labeled data
6. **Replace default YOLO** with your custom model

---

**Need help?** See `README.md` for full documentation.

Happy counting! 🐛
