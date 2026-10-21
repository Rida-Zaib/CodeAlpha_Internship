---
title: The Royal Court of Vision
emoji: 👑
colorFrom: teal
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# 👑 The Royal Court of Vision

**Task 4 - Object Detection and Tracking.** A 3D invitation-style website backed by a real computer-vision API (FastAPI + OpenCV + YOLO).

| Task requirement | Where it is |
|---|---|
| Real-time video input (webcam / video file) with OpenCV | `object_tracking.py` (desktop window) and the website's **Live Camera** and **Video / Image upload** modes (`backend/main.py`) |
| Pre-trained detector | YOLOv8 / YOLO11 (COCO, 80 classes) and YOLO-World (about 2,000 names) via Ultralytics - `backend/engine.py` |
| Boxes on every frame | `TrackingSession.draw()` - corner-bracket boxes, labels, motion trails, tripwire |
| SORT tracking (Deep SORT not included) | `backend/sort_tracker.py` - own Kalman filter + Hungarian algorithm; ByteTrack and BoT-SORT via Ultralytics |
| Labels + tracking IDs in real time | `ID 7 | car 0.93` on every box; live counters for line crossings |

## 1. Install everything FIRST (once, before any testing)

Requirements: **Python 3.10 - 3.12**, about **3 GB** free disk, and an internet connection for this step only.

### Windows (PowerShell)
```powershell
cd royal-court-of-vision-full
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/ultralytics/CLIP.git      # needs git; if git is missing use:  pip install ftfy regex
cd backend
python prepare_models.py --all --smoke                         # downloads every model (about 2 GB, 10-30 min) and runs one test photo
```

### macOS / Linux
```bash
cd royal-court-of-vision-full
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/ultralytics/CLIP.git      # needs git; if git is missing use:  pip install ftfy regex
cd backend
python prepare_models.py --all --smoke
```

Useful: `python prepare_models.py --check` shows what is already downloaded (it downloads nothing). `python prepare_models.py` (without `--all`) downloads only the essentials.
Models are stored in `backend/models`, `backend/weights` and the Hugging Face cache. Copy those folders to another computer to skip the download there.

## 2. Run the website
```powershell
cd backend
uvicorn main:app --port 8000
```
Open **http://localhost:8000** (it must be `localhost`; browsers block the webcam on other plain-http addresses). The backend serves the website too. API docs: http://localhost:8000/docs.
Stop the server with `Ctrl + C`.

How to use it:
1. **Video / Image** tab: upload a file, pick a model (YOLOv8n is fastest) and press start. A video is processed once, completely, and plays once.
2. **Live Camera** tab: allow the camera and press start.
3. Leave **Detection Targets** on *Auto*, or switch to *Choose targets* and pick from the menus.

## 3. Run the desktop real-time window (no browser)
```powershell
python object_tracking.py                                                    # webcam, press q or ESC to quit
python object_tracking.py --source video.mp4 --save out.mp4 --tracker bytetrack
```
Run it from the project folder (the one that contains `object_tracking.py`), with the virtual environment active.

## 4. Run the tests (optional, no model download needed)
```powershell
pip install pytest httpx
set RCV_FAKE_DETECTOR=1          # PowerShell:  $env:RCV_FAKE_DETECTOR="1"      macOS/Linux:  export RCV_FAKE_DETECTOR=1
pytest -q
```
`RCV_FAKE_DETECTOR=1` replaces YOLO with a synthetic detector so the tracker, counter, video/heatmap/CSV writer and the API can be tested offline.

## 5. Troubleshooting
| Problem | Fix |
|---|---|
| `uvicorn` is not recognised | The virtual environment is not active. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside the active environment. |
| Webcam does not start | Open `http://localhost:8000` (not an IP address or a file), allow the camera and close other apps that use it. |
| First image in Auto mode is slow or uses the standard detector | The wide-vocabulary model is still preparing. Run `python prepare_models.py --all` once and wait for it to finish. |
| Port 8000 is busy | Use `uvicorn main:app --port 8001` and open `http://localhost:8001`. |
| Slow video | Use YOLOv8n, detection resolution 640 and a clip of 5-10 seconds while testing. |

## Reading the results table
| Column | Meaning |
|---|---|
| Frame | Video frame number of the row (frame 1 = first image). |
| Track ID | One number per physical object, kept while it is followed. |
| Class | What the detector thinks it is; a finer vehicle type or model may be shown below it. |
| Conf | Detector confidence from 0 to 1 (lower values are more likely false detections). |
| Centroid | (x, y) pixel position of the box centre; (0, 0) is the top-left corner and y grows downward. |
| Crossed | Whether the object has crossed the virtual tripwire line. |
| Direction | DOWN or UP when it crossed the line; NEW when first seen. |

The downloaded CSV also has the box corners (x1, y1, x2, y2), a *heading* column (left, right, up, down, stationary, new) and an *attributes* column.

## Configuration (environment variables)
`RCV_MAX_UPLOAD_MB` (120) · `RCV_MAX_WIDTH` (960) · `RCV_WORKERS` (1) · `RCV_JOB_TTL_S` (3600) · `RCV_AUTO_FLOOR` (0.40, minimum score shown in Auto mode) · `RCV_BLUR_BELOW` (0.22, blur filter) · `RCV_NO_WARMUP=1` (do not download models on start-up).

## Folder layout
```
backend/   main.py (API)  engine.py (YOLO + tracking + drawing)  sort_tracker.py  vehicles.py  verify.py  prepare_models.py
frontend/  index.html (website)  assets/  favicon.*
object_tracking.py   desktop real-time version
Dockerfile · requirements.txt · tests/
```

## Deploy for free - Hugging Face Spaces
Create a free Space (SDK **Docker**, CPU basic), upload this folder, wait 5-10 minutes for the build. Keep clips to 10-20 s on the free CPU.

## Licence note
Ultralytics YOLO is AGPL-3.0: free for study, research and portfolio projects. A commercial product needs an Ultralytics licence.

## Known limitations (read before judging the results)

* **Everything is free and runs locally.** No API key, no paid service. Models are downloaded once from public sources.
* **Accuracy depends on the model size.** The default YOLOv8n (6 MB) is the fastest but misses small or distant objects. Pick YOLOv8s / YOLO11s
  (about 20 MB) or YOLOv8m (about 50 MB) in the checkpoint menu, and 960 px detection resolution, for better results at the cost of speed.
* **Auto mode (about 2,000 names) is an estimate.** The wide-vocabulary detector can mislabel unusual scenes. Out-of-focus blobs are filtered and names below a
  0.40 score are hidden, but errors can still happen. Raise the Confidence slider for stricter results.
* **Vehicle types and models** are CLIP zero-shot guesses, not a trained car-model recogniser. Types (hatchback, scooter, tempo) are usually right; exact
  models (e.g. Swift, Splendor) are often wrong on small or blurry vehicles. They are available for images and video files only.
* **Tracking:** SORT and ByteTrack (no Deep SORT). IDs can switch when objects are hidden behind each other for a long time.
* **Live webcam** works in the browser at localhost. **RTSP / IP camera is marked "coming soon"** in the website and is not part of the tested feature set (the code exists in `backend/streams.py`).
* **Traffic Survey Report** is available for uploaded videos only.
* **Videos** are processed once, completely; long videos take proportionally longer. CPU-only machines are slow with the larger models.

### Free upgrades planned for the future
Larger free detectors (YOLOv8l/x, RT-DETR), Deep SORT / BoT-SORT appearance tracking, a free local vision-language model (e.g. Qwen2-VL, SmolVLM) for reviewing
doubtful detections, a fine-tuned vehicle-model classifier on a public dataset (Stanford Cars / CompCars), and live-mode vehicle recognition.