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

**Task 4 - Object Detection and Tracking.** A 3D invitation-style website backed by a real computer-vision API.

| Task requirement | Where it is |
|---|---|
| Real-time video input (webcam / video file) with OpenCV | `object_tracking.py` (desktop window) and the website's **Live Camera**, **File** and **RTSP** modes (`backend/main.py`, `backend/streams.py`) |
| Pre-trained detector | YOLOv8n / YOLOv8s (COCO, 80 classes) via Ultralytics - `backend/engine.py` |
| Boxes on every frame | `TrackingSession.draw()` - corner-bracket boxes, labels, motion trails, tripwire |
| SORT tracking (Deep SORT not included) | `backend/sort_tracker.py` - own Kalman filter + Hungarian algorithm; ByteTrack and BoT-SORT via Ultralytics |
| Labels + tracking IDs in real time | `ID 7 \| car 0.93` on every box; live counters for line crossings |

Options: 5 models (YOLOv8n/s/m, YOLO11n/s), detection resolution 480-1280 px, 3 trackers.

### Detection targets (new)
* **Auto** (default) - nothing to choose. Image mode names whatever is in the picture from a vocabulary of about 340 everyday things (lion, pigeon, whiteboard, ...) using YOLO-World; video / live use all 80 COCO classes.
* **Choose targets** - nine themed multi-select dropdown menus (People & Vehicles, Animals, Carried Items on the first row; Clothing, Street, Sports & Toys, Food & Kitchen, Home, Electronics below). Picks appear as removable tags. The filter is applied *after* detection, so a lion is never forced into "dog".
* The menus are generated from `backend/vocab.py` (`python tools/gen_class_menus.py` after editing it).

### No waiting for downloads
The wide-vocabulary model (YOLO-World + CLIP, ~370 MB) is downloaded **once, in the background, when the server starts**, then baked into `backend/models/universal-world-<N>.pt` (re-baked automatically whenever the vocabulary changes). Until it is ready the site stays usable (standard 80-class detector, with a note) and shows the progress. Models are always stored under `backend/` no matter which folder you start the server from. To prepare everything up front: `cd backend && python prepare_models.py` (the CLIP model for vehicle types / models and person details is included; add `--no-people` to skip it, ~600 MB). The Dockerfile does this while building, so deployed visitors never wait.

### Smart Vision (beta)
| Option | What it does | Extra download (first use) |
|---|---|---|
| Open vocabulary | type any object names (pencil, whiteboard, shopping bag, ...) or tap the Classroom / Mall / Street / Playground packs; uses YOLO-World | YOLO-World weights + CLIP text encoder |
| Tiny-object scan | image is split into overlapping tiles and merged with NMS, so very small objects are found | none (slower) |
| Analyze people | per person: man / woman / boy / girl / teen / elderly, bag, phone use, reading / writing / playing / walking... (CLIP zero-shot on each person crop) | about 600 MB (`openai/clip-vit-base-patch32`) |

People attributes are *appearance-based estimates*, not ground truth: they can be wrong for small, blurry or partly hidden people. Detection itself stays real-time; attributes are computed once per person track.

Extras: **image detection**, line-crossing counter, movement heatmap, CSV log, per-class counts, auto API docs at `/docs`.

## Input modes on the website
1. **File** - upload a video, get the tracked MP4, heatmap, CSV and counters (background job + progress bar).
2. **Image** - upload a photo, get detections, density heatmap and CSV.
3. **Live Camera** - browser webcam frames go to the server over a WebSocket; tracks come back and are drawn live.
4. **RTSP / IP camera** - the server reads the stream, tracks it and sends back an annotated MJPEG stream.

If the server is unreachable, the site falls back to a built-in simulation so it never looks broken (status badge shows *Demo mode*).

## Run locally (Windows / macOS / Linux)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cd backend
uvicorn main:app --port 8000
```
Open **http://localhost:8000** - the backend serves the website too. API docs: http://localhost:8000/docs.
The first run downloads `yolov8n.pt` (6 MB) automatically.

Desktop real-time window (no browser):
```bash
python object_tracking.py                                  # webcam
python object_tracking.py --source video.mp4 --save out.mp4 --tracker bytetrack
```

## Deploy for free (one link, everything included) - Hugging Face Spaces
1. Create a free account at huggingface.co -> **New Space** -> SDK **Docker**, hardware **CPU basic (free)**.
2. Upload every file of this folder (or `git push` to the Space). The YAML header above is already in this README.
3. Wait for the build (about 5-10 minutes). Your site is live at `https://<user>-<space>.hf.space` - website and API on the same URL, nothing to configure.

Notes: the free CPU is slow for long videos, so keep clips to 10-20 s. The Space sleeps when idle; the first request can take 30-60 s. Browsers only allow webcam on `https://` or `localhost` (Spaces is https, so it works).

### Optional: website and API on separate hosts
The site talks to the server it is served from, so no URL box is needed. If you host `frontend/` elsewhere (Netlify / GitHub Pages), open it once as `https://your-site/?api=https://<user>-<space>.hf.space` - the address is remembered.

## Tests
```bash
set RCV_FAKE_DETECTOR=1        # Windows (Linux/macOS: export RCV_FAKE_DETECTOR=1)
pip install pytest httpx
pytest -q
```
`RCV_FAKE_DETECTOR=1` replaces YOLO with a synthetic detector so the tracker, counter, video/heatmap/CSV writer and the API can be tested offline.

## Configuration (environment variables)
`RCV_MAX_UPLOAD_MB` (120) · `RCV_MAX_WIDTH` (960, frames are downscaled for speed) · `RCV_WORKERS` (1) · `RCV_MAX_STREAMS` (2) · `RCV_JOB_TTL_S` (3600).

## Folder layout
```
backend/   main.py (API)  engine.py (YOLO + tracking + drawing)  sort_tracker.py  streams.py (RTSP)
frontend/  index.html (website)  assets/  favicon.*
object_tracking.py   desktop real-time version
Dockerfile · requirements.txt · tests/
```

## Licence note
Ultralytics YOLO is AGPL-3.0: free for study, research and portfolio projects. A commercial product needs an Ultralytics licence.


## Video: one full pass, then stop (v1.1)

* The whole video is processed **once, every frame, whatever its length**; detection stops when the video ends. The result player no longer loops or autoplays.
* After detection a clean-up pass runs over the *whole* clip before anything is drawn: flicker (tracks under ~0.25 s or never clearly confident) is dropped, two tracks sitting on the same box (e.g. one car reported as `truck` *and* `car`) are merged into one object, and each object gets a single class decided by a confidence-weighted vote over its whole life (a mild prior prefers `car` when the detector hesitates between car and truck). Ids are renumbered 1, 2, 3 ... in order of appearance.
* Image mode and live webcam / RTSP also keep only one box per object.

## Vehicle types and models (free format)

Tick **Identify vehicle types & models** (on by default). For every car, truck, bus, motorcycle, bicycle (and train / boat / aircraft) the best crops are compared with CLIP against about 1,000 names: types (hatchback, SUV, tempo, auto rickshaw, dump truck, mountain bike, cruiser motorcycle ...) and makes / models (Maruti Suzuki Swift, Tata Ace, Bajaj Maxima, Royal Enfield Classic 350, Hero Splendor, Trek Marlin ...). The result appears under the class name in the table, on the video and in the CSV (`attributes` column). If CLIP strongly disagrees with YOLO on the broad class, the class is corrected.
Anything you type in **Find anything by name** is added to the candidates for that run, so you can search for names that are not built in. These are appearance-based estimates: types are usually right, an exact model is shown only when the match is clear, and small or blurry vehicles fall back to the type (or no label).

## Vocabulary

About 2,160 everyday things in 12 menus (people, vehicles, animals, carried items, clothing, street, sports and toys, food and kitchen, home, electronics and office, tools, music and art, plants and nature), all used by Auto mode on images. Edit `backend/vocab_big.py` / `backend/vehicles.py`, then run `python tools/gen_class_menus.py`.


## Set up everything BEFORE testing (no downloads while you wait)

```bash
# 1. once: environment + packages
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision                              # CPU is fine; add --index-url https://download.pytorch.org/whl/cpu for the small build
pip install -r requirements.txt
pip install git+https://github.com/ultralytics/CLIP.git   # needs git; used by the wide vocabulary (falls back to: pip install ftfy regex)

# 2. once: download every model (about 2 GB, 10-30 min depending on your connection)
cd backend
python prepare_models.py --all --smoke                     # --check shows what is already there, downloads nothing

# 3. every time: start the site (nothing left to download)
uvicorn main:app --port 8000                               # open http://localhost:8000
```

`--all` covers every detector in the model menu (YOLOv8n/s/m, YOLO11n/s, YOLO-World s/m), the wide-vocabulary model and the CLIP model that powers vehicle types / models and "Analyze people". Webcam and RTSP use the same detectors, so they need nothing extra. ffmpeg for the result video comes with `imageio-ffmpeg` (in requirements). Models live in `backend/models` and `backend/weights`, plus the Hugging Face cache; copy those folders to another machine to skip the download there.

### Faster test runs
* Use `yolov8n`, detection resolution 640, and a clip of 5-10 seconds while testing.
* Leave "Analyze people" off unless you are testing it; keep "Identify vehicle types & models" on only when vehicles matter.
* Use "Choose targets" with a few classes instead of Auto for videos when you only need those.
* On CPU a video costs roughly (frames x ~0.1-0.3 s); a GPU build of PyTorch makes it several times faster. `RCV_MAX_WIDTH=640` downsizes big videos before detection.
* Run `pytest -q` (no downloads, fake detector) for logic checks; use the real run only for the final check.


## Accuracy of Auto (v1.2): detector proposes, CLIP names

YOLO-World finds things but names them badly among thousands of words (squirrel -> "beaver", mushroom -> "tree trunk"). In Auto / typed-name / wide-vocabulary runs every box is now cropped and re-named by CLIP (`backend/verify.py`); boxes CLIP calls "background / blurry foliage / texture" or is unsure about are dropped, and duplicates are merged. Scenery with no clear outline (forest, jungle, sky, walls, roads ...) is no longer searched in Auto (still in the menus - picking one builds its own search). Tunable with `RCV_VERIFY_ACCEPT` (default 0.35, raise for fewer, surer boxes) and `RCV_VERIFY_AGREE` (0.15). Until the CLIP model has downloaded, Auto runs strict (confidence 0.45) and says so.

## Traffic Survey Report (v1.3)

After every video the page shows a **Traffic Survey Report** and offers it as a printable page (`/api/jobs/<id>/report`, print to PDF) and a CSV (`/report.csv`):
counts by kind (pedestrians, cars, trucks / tempos, buses, motorcycles, bicycles) and direction (down / up across the counting line), a table per time slot with the busiest slot, the vehicle types and models seen, vehicles that **stood still** for a chosen time (parked / blocked lane) and vehicles **going against an expected direction**. Settings are next to "Processing Horizon": time slot (auto = 10 s / 1 min / 5 min / 15 min by video length), expected direction, and the stop time to flag.
Notes: the counting line is horizontal, so directions are "down" and "up" on screen; mount the camera so the road runs across the picture height for best counts. Counts are automatic estimates - compare a few minutes with a manual count. Stopped vehicles include red-light and jam stops. Code: `backend/report.py`.

## Known limitations (read before judging the results)

* **Everything is free and runs locally.** No API key, no paid service. Models are downloaded once from public sources.
* **Accuracy depends on the model size.** The default YOLOv8n (6 MB) is the fastest but misses small or distant objects. Pick YOLOv8s / YOLO11s
  (about 20 MB) or YOLOv8m (about 50 MB) in the checkpoint menu, and 960 px detection resolution, for better results at the cost of speed.
* **Auto mode (about 2,000 names) is an estimate.** The wide-vocabulary detector can mislabel unusual scenes. Out-of-focus blobs are filtered and names below a
  0.40 score are hidden, but errors can still happen. Raise the Confidence slider for stricter results.
* **Vehicle types and models** are CLIP zero-shot guesses, not a trained car-model recogniser. Types (hatchback, scooter, tempo) are usually right; exact
  models (e.g. Swift, Splendor) are often wrong on small or blurry vehicles. They are available for images and video files only.
* **Tracking:** SORT and ByteTrack (no Deep SORT). IDs can switch when objects are hidden behind each other for a long time.
* **Live webcam** works in the browser at localhost. **RTSP / IP camera is marked "coming soon"** and is not part of the tested feature set.
* **Traffic Survey Report** is available for uploaded videos only.
* **Videos** are processed once, completely; long videos take proportionally longer. CPU-only machines are slow with the larger models.

### Free upgrades planned for the future
Larger free detectors (YOLOv8l/x, RT-DETR), Deep SORT / BoT-SORT appearance tracking, a free local vision-language model (e.g. Qwen2-VL, SmolVLM) for reviewing
doubtful detections, a fine-tuned vehicle-model classifier on a public dataset (Stanford Cars / CompCars), and live-mode vehicle recognition.
