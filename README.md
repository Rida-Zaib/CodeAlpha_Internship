# CodeAlpha Internship Projects

Two projects built for the **CodeAlpha** internship. Each one lives in its own folder with its own dependencies, so you can set up and run them separately.

| | Project | What it does | Main tech |
|---|---|---|---|
| **Task 1** | [TechNest AI Support: FAQ Chatbot](#task-1--technest-ai-support-faq-chatbot) | Answers customer questions from an FAQ list, however they are worded, with an optional free AI mode | Python, Flask, NLTK, TF-IDF, free LLM APIs (Groq / Gemini / OpenRouter) |
| **Task 4** | [The Royal Court of Vision: Object Detection and Tracking](#task-4--the-royal-court-of-vision-object-detection-and-tracking) | Detects objects in videos, photos and webcam, tracks each one with an ID, and builds traffic reports | Python, OpenCV, YOLO, CLIP, FastAPI, Three.js |

## Repository layout

```
CodeAlpha-Internship/
├── README.md                         this file
├── Task1_FAQ_Chatbot/                the FAQ chatbot (Flask app)
│   ├── faqs.csv  preprocess.py  matcher.py  app.py  chatbot_cli.py  gap_report.py
│   ├── templates/index.html  tests/  data/  requirements.txt
└── Task4_Object_Detection/           the detection and tracking app
    ├── backend/    main.py  engine.py  sort_tracker.py  vehicles.py  verify.py  prepare_models.py
    ├── frontend/   index.html  assets/
    ├── object_tracking.py            desktop real-time window
    └── Dockerfile  requirements.txt  tests/
```

Requirements for both: **Python 3.10 to 3.12**, `git`, and an internet connection for the first install. Task 4 needs about 3 GB of free disk space for its models.

---

# Task 1 · TechNest AI Support: FAQ Chatbot

A customer-support chatbot that understands questions however they are worded: short, long, casual, with typos, or several questions in one message.

**Two modes:**

* **Smart mode** (needs one free API key): an LLM reads the customer's message together with the FAQ(s) that were found, and phrases the answer naturally. It remembers the last few messages, so follow-ups like "and for international?" work. It is told to answer only from the FAQ and to say so when the FAQ does not cover something.
* **Local matcher** (no key, no internet): classic NLP (NLTK and TF-IDF / cosine similarity) over the FAQs and their alternate phrasings. Instant, free, and used automatically whenever Smart mode is off or unreachable.

## 1. Setup

```bash
cd Task1_FAQ_Chatbot
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Download the NLTK data used for preprocessing (one-time):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

If you skip this, the bot still runs. `preprocess.py` falls back to a built-in stopword list instead of crashing, but matching is a bit sharper with the real NLTK data (proper lemmatization: "returning", "returns" and "returned" all collapse to one word). When the app starts, the terminal prints whether NLTK is really being used. `GET /health` shows the same under `nlp`.

## 2. Run it

```bash
python app.py                  # web chat UI -> http://127.0.0.1:5000
python chatbot_cli.py          # terminal version
pip install pytest && pytest tests/ -v    # optional tests
```

## 3. How it works

* **`faqs.csv`**: 25 Q&A pairs for a fictional electronics store, each tagged with a category (Shipping, Returns, Payments, Warranty, Account, Support).
* **`preprocess.py`**: lowercases, strips punctuation, tokenizes, removes stopwords, lemmatizes.
* **`matcher.py`** is the local engine. For every message it:
  * indexes each FAQ under its question plus the alternate phrasings in the `alternates` column (including Roman Urdu), and also searches the answer text;
  * splits long messages into clauses so filler words do not drown the real question, and answers every part of a multi-part question;
  * expands synonyms ("money back" becomes "refund"), recognises sentence shapes ("when will my stuff show up" means delivery time) and corrects typos;
  * says so honestly when nothing matches, offers the closest questions it can answer, and logs the question to `data/unanswered.csv`. Run `python gap_report.py` to see which FAQs or alternates to add next.
* **`app.py`** exposes:
  * `POST /ask` with `{message, history, mode}`, returning `{answer, answered, suggestions, ...}`. Greetings and thanks are answered instantly without using any AI quota.
  * `POST /feedback` logs thumbs up/down to `data/feedback.csv`, so you can see which answers were wrong.
  * `GET /health` returns status, number of FAQs, and which AI providers are configured.

## 4. Smart mode: AI-phrased answers (optional)

By default the bot answers purely from TF-IDF matching, which needs no setup and never calls the internet. Smart mode sends the FAQ(s) that TF-IDF already retrieved to a free LLM API, which rephrases the answer more naturally. The LLM is never allowed to answer from its own knowledge, only from the FAQ context it is given, so it cannot invent policies or numbers that are not in `faqs.csv`. This is retrieval-augmented generation (RAG), not fine-tuning.

**Setup (optional):**

1. Copy `.env.example` to `.env`.
2. Get a free API key from one or more of these (no credit card needed):
   * **Groq**: console.groq.com/keys → `GROQ_API_KEY` (set this one up first)
   * **Google Gemini**: aistudio.google.com/apikey → `GEMINI_API_KEY`
   * **OpenRouter**: openrouter.ai/keys → `OPENROUTER_API_KEY`
   * **Hugging Face** (optional, least reliable): huggingface.co/settings/tokens → `HF_TOKEN`. Its free allowance is small and many accounts get HTTP 402 or 400 straight away. The app detects that, says so once, and stops trying for 30 minutes.
3. Paste the key(s) into `.env`. One is enough; two to four give automatic fallback.
4. Restart `python app.py`.

**Good to know:**

* **Models get retired and the app copes.** For Groq, Gemini and OpenRouter the app tries its current models in order, remembers any the provider says are gone, and if all are gone it asks the provider which models it offers today. You can pin your own with `GROQ_MODEL=` and similar in `.env`.
* **Racing, not one-at-a-time.** On every Smart-mode request the app calls every configured provider at the same time on separate threads. Whichever answers first wins. If none respond within 20 seconds, it silently falls back to the instant TF-IDF answer, so the chatbot never breaks because of this.
* **Junk is not sent to the AI.** Smart mode only calls the LLM when TF-IDF already found a relevant FAQ. Gibberish goes straight to the honest "I don't know" reply, so no quota is wasted.
* **Persistent answer cache** (`ai_cache.py`). Smart-mode answers are cached in `data/ai_cache.json`, keyed by the exact question (ignoring case and punctuation). Follow-ups are not cached. Delete this file after editing `faqs.csv`.
* **If answers look like copied FAQ text**, the AI is not working. The header then says "AI offline, using basic answers". Run `python check_ai.py`. It makes one real request per key and prints the exact reason (rejected key, rate limit, no internet). `GET /health` also shows it under `ai_errors`.

## 5. What the UI shows

A deliberately plain chat: the answer, and nothing else cluttering it.

* Starter chips with a few example questions.
* Thumbs up / down under every real answer, feeding `data/feedback.csv`.
* "You could try asking:" chips, only when the bot could not answer.
* A Smart / Fast switch in the header, shown only when an AI key is configured.
* A New conversation button that clears the chat and its memory.

## 6. Using your own FAQs

Edit `faqs.csv`. Keep the header `category,question,answer,alternates`. The `alternates` column is optional: other ways people ask the same thing, separated by `|`. If your customers use wording your FAQ does not (for example "cancel my purchase" versus "cancel order"), add that pair to the `SYNONYMS` dict at the top of `matcher.py`. The index is rebuilt from the CSV on startup, so no other code changes are needed.

## 7. Put it on GitHub

Hosting platforms deploy from a GitHub repository, so do this first.

1. Install Git from git-scm.com (check with `git --version`), then tell it who you are:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
2. On github.com click **New repository**, give it a name (for example `CodeAlpha-Internship`), and leave README, .gitignore and license **unticked**.
3. In a terminal in the repository folder (the one that holds both task folders):
   ```bash
   git init
   git add .
   git status
   ```
   Read the list. **`.env` must not be in it** (it holds your API keys; `.gitignore` keeps it out). `.env.example` should be there.
4. Then:
   ```bash
   git commit -m "CodeAlpha projects"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
   The first push opens a browser to sign in to GitHub.
5. If a key was ever committed by mistake, delete that key at the provider and create a new one. Removing the file later does not remove it from the history.

Later changes: `git add .`, `git commit -m "what changed"`, `git push`.

## 8. Deploy it (free)

The backend and frontend deploy together as one service. `app.py` serves both the chat page and the `/ask`, `/health` and `/feedback` APIs, so there is one live URL and no CORS setup.

**Why not Vercel:** it runs Python as short-lived serverless functions where nothing written to disk survives between requests. That breaks the on-disk answer cache and the startup warm-up.

### Recommended: Render

Free plan (as of Oct 2026, check render.com/pricing): 512 MB RAM, a small share of a CPU, 750 free hours a month, no persistent disk. It may ask for a payment method when you create the service; free usage is not billed.

1. Push the project to GitHub.
2. render.com → sign up with GitHub → **New +** → **Web Service** → pick your repo.
3. Fill in:
   * **Root Directory:** `Task1_FAQ_Chatbot`
   * **Region:** nearest one (Singapore for Pakistan). **Branch:** `main`. **Runtime:** Python 3
   * **Build Command:** `pip install -r requirements.txt && python setup_nltk.py`
   * **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 90`
   * **Instance Type:** Free
   * **Environment Variables:** the same names as your `.env`, for example `GROQ_API_KEY`. Skip `HF_TOKEN`.
4. Create the service and watch the logs. A good run ends with `Loaded 25 FAQs`, `Text preprocessing: NLTK active` and `Your service is live`.
5. Open `https://your-name.onrender.com/health`. `ai_providers_configured` should list `groq`.

Every `git push` to `main` redeploys automatically.

**Free plan behaviour:** after 15 minutes without visitors it sleeps and the next visit takes about a minute. Files written at runtime (cache, feedback, unanswered logs) are lost on restart. Anyone with the link can use your AI key's free quota, so share it with care.

**Troubleshooting:**

* Build fails installing scikit-learn / numpy: add the environment variable `PYTHON_VERSION = 3.12.8` and redeploy.
* "No open ports detected": the Start Command must contain `--bind 0.0.0.0:$PORT` exactly.
* "NLTK data missing": the Build Command is missing `&& python setup_nltk.py`.
* Header says "AI offline": open `/health` and check `ai_errors`, or the key's name and value under Environment.
* "Out of memory": keep `--workers 1`.

### No card? Use PythonAnywhere

The free Beginner account needs no payment method. Its free plan only lets the app reach an allowlist of internet sites, which (checked Oct 2026) includes `api.groq.com` and `*.googleapis.com` (Gemini), so use a Groq or Gemini key. Free limits: 512 MB disk, 1 web app with 1 worker, and the web app expires after 1 month unless you click the extend button.

1. Create a free Beginner account at pythonanywhere.com.
2. **Consoles** tab → start a Bash console and get the code (the GitHub repo must be Public, which is safe because `.env` is never uploaded):
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   ```
3. Make a virtualenv and install:
   ```bash
   mkvirtualenv --python=python3.12 --system-site-packages faqbot
   cd ~/YOUR_REPO/Task1_FAQ_Chatbot
   pip install -r requirements.txt
   python setup_nltk.py
   ```
   If `python3.12` is not found, try `python3.11` or `python3.10`.
4. Create your keys file: `nano .env`, type `GROQ_API_KEY=your_key`, then Ctrl+O, Enter, Ctrl+X. Test with `python check_ai.py`.
5. **Web** tab → **Add a new web app** → **Manual configuration** → pick the same Python version.
6. Set **Source code and Working directory** to `/home/USERNAME/YOUR_REPO/Task1_FAQ_Chatbot` and **Virtualenv** to `/home/USERNAME/.virtualenvs/faqbot`.
7. Open the WSGI configuration file, delete everything, and paste (replace `USERNAME` and `YOUR_REPO`):
   ```python
   import os, sys
   path = "/home/USERNAME/YOUR_REPO/Task1_FAQ_Chatbot"
   if path not in sys.path:
       sys.path.insert(0, path)
   os.chdir(path)   # the app reads faqs.csv and writes data/ relative to this folder
   from app import app as application
   ```
8. Click **Reload**, then open `https://USERNAME.pythonanywhere.com/health`. Every month, use the Web tab button that extends the app's life. To update later: `git pull`, then Reload.

If something fails, check the Web tab → Error log. "Disk quota exceeded": delete the virtualenv (`rmvirtualenv faqbot`) and redo step 3 with `--system-site-packages`.

## 9. Known limitation (Task 1)

TF-IDF and keyword overlap match on words, not deep meaning. The synonym map and typo correction close the most common gaps, but a truly novel phrasing may still miss. For stronger semantic matching you would move to sentence embeddings (for example `sentence-transformers`), a heavier dependency than this project needs. `gap_report.py` shows exactly which real questions are being missed so you can patch `SYNONYMS` or `faqs.csv`.

---

# Task 4 · The Royal Court of Vision: Object Detection and Tracking

A 3D invitation-style website backed by a real computer-vision API (FastAPI, OpenCV and YOLO). Upload a video or photo, or use your webcam, and it detects objects, follows each one with a tracking ID, and builds a traffic survey report.

| Task requirement | Where it is |
|---|---|
| Real-time video input (webcam or video file) with OpenCV | `object_tracking.py` (desktop window) and the website's **Live Camera** and **Video / Image upload** modes (`backend/main.py`) |
| Pre-trained detector | YOLOv8 / YOLO11 (COCO, 80 classes) and YOLO-World (about 2,000 names) through Ultralytics, in `backend/engine.py` |
| Bounding boxes on every frame | `TrackingSession.draw()`: corner-bracket boxes, labels, motion trails and the tripwire |
| SORT tracking (Deep SORT is not included) | `backend/sort_tracker.py`: own Kalman filter and Hungarian algorithm. ByteTrack and BoT-SORT through Ultralytics |
| Labels and tracking IDs in real time | `ID 7 \| car 0.93` on every box, with live counters for line crossings |

**What it does:**

* Detects objects with YOLOv8 / YOLO11, and names about 2,000 kinds of things in Auto mode (YOLO-World).
* Tracks each object with SORT, ByteTrack or BoT-SORT, and draws the label and ID on every frame.
* Counts vehicles crossing a virtual tripwire, and builds a traffic survey report (time slots, wrong-way direction, stopped vehicles) for videos.
* Guesses vehicle types and models with CLIP, and draws a heatmap of where detections cluster.
* Filters false detections: out-of-focus blobs are ignored, weak names are hidden, and duplicate boxes on one object are merged.

**Architecture:**

```
Browser (HTML / CSS / JS + Three.js 3D scene)
   |  upload video / photo, or webcam
   v
FastAPI backend (backend/main.py)
   |-- Detection ........ YOLOv8 / YOLO11, YOLO-World (open vocabulary)
   |-- Verification ..... CLIP name check, vehicle types, sharpness filter
   |-- Tracking ......... SORT, ByteTrack, BoT-SORT
   |-- Analytics ........ tripwire counts, traffic report, heatmap, CSV
   v
Annotated video / image + report + CSV
```

## 1. Install everything first (once, before any testing)

Requirements: about **3 GB** free disk and an internet connection for this step only.

**Windows (PowerShell):**

```powershell
cd Task4_Object_Detection
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/ultralytics/CLIP.git      # needs git; if git is missing use:  pip install ftfy regex
cd backend
python prepare_models.py --all --smoke                         # downloads every model (about 2 GB, 10-30 min) and runs one test photo
```

**macOS / Linux:** the same, but use `python3 -m venv .venv` and `source .venv/bin/activate`.

`python prepare_models.py --check` shows what is already downloaded (it downloads nothing). Models are stored in `backend/models`, `backend/weights` and the Hugging Face cache. Copy those folders to another computer to skip the download there.

## 2. Run the website

```powershell
cd backend
uvicorn main:app --port 8000
```

Open **http://localhost:8000**. It must be `localhost`, because browsers block the webcam on other plain-http addresses. API docs are at http://localhost:8000/docs. Stop the server with `Ctrl + C`.

How to use it:

1. **Video / Image** tab: upload a file, pick a model (YOLOv8n is fastest) and press start. A video is processed once, completely, and plays once.
2. **Live Camera** tab: allow the camera and press start.
3. Leave **Detection Targets** on *Auto*, or switch to *Choose targets* and pick from the menus.

## 3. Run the desktop real-time window (no browser)

From the Task 4 folder, with the environment active:

```powershell
python object_tracking.py                                                    # webcam, press q or ESC to quit
python object_tracking.py --source video.mp4 --save out.mp4 --tracker bytetrack
```

## 4. Run the tests (optional, no model download needed)

```powershell
pip install pytest httpx
$env:RCV_FAKE_DETECTOR="1"          # macOS/Linux:  export RCV_FAKE_DETECTOR=1
pytest -q
```

`RCV_FAKE_DETECTOR=1` replaces YOLO with a synthetic detector so the tracker, counter, video / heatmap / CSV writer and the API can be tested offline.

## 5. Reading the results table

| Column | Meaning |
|---|---|
| Frame | Video frame number of the row (frame 1 is the first image). |
| Track ID | One number per physical object, kept while it is followed. |
| Class | What the detector thinks it is. A finer vehicle type or model may be shown below it. |
| Conf | Detector confidence from 0 to 1. Lower values are more likely false detections. |
| Centroid | (x, y) pixel position of the box centre. (0, 0) is the top-left corner and y grows downward. |
| Crossed | Whether the object has crossed the virtual tripwire line. |
| Direction | DOWN or UP when it crossed the line, NEW when first seen. |
| Size, Region | Images only: box width × height in pixels, and which part of the picture it sits in. |

The downloaded CSV also has the box corners (x1, y1, x2, y2), a heading column (left, right, up, down, stationary, new) and an attributes column. The ledger has one row per tracked object.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` is not recognised | The virtual environment is not active. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside the active environment. |
| Webcam does not start | Open `http://localhost:8000` (not an IP address or a file), allow the camera, and close other apps that use it. |
| First image in Auto mode is slow, or uses the standard detector | The wide-vocabulary model is still preparing. Run `python prepare_models.py --all` once and wait for it to finish. |
| Port 8000 is busy | Use `uvicorn main:app --port 8001` and open `http://localhost:8001`. |
| Slow video | Use YOLOv8n, detection resolution 640 and a clip of 5 to 10 seconds while testing. |

## 7. Configuration (environment variables)

`RCV_MAX_UPLOAD_MB` (120) · `RCV_MAX_WIDTH` (960) · `RCV_WORKERS` (1) · `RCV_JOB_TTL_S` (3600) · `RCV_AUTO_FLOOR` (0.40, minimum score shown in Auto mode) · `RCV_BLUR_BELOW` (0.22, blur filter) · `RCV_NO_WARMUP=1` (do not download models on start-up).

## 8. Deploy for free: Hugging Face Spaces

Create a Space with the **Docker** SDK on the free CPU Basic hardware (2 vCPU, 16 GB RAM, which suits the models). Upload the Task 4 folder, which already contains a `Dockerfile`, and wait 5 to 10 minutes for the build. Free Spaces sleep when unused, and sources disagree on whether a new account needs a paid plan to create a compute Space, so check huggingface.co/new-space first. Keep clips to 10 to 20 seconds on the free CPU.

## 9. Known limitations (Task 4)

* **Everything is free and runs locally.** No API key and no paid service. Models are downloaded once from public sources.
* **Accuracy depends on the model size.** The default YOLOv8n (6 MB) is the fastest but misses small or distant objects. Pick YOLOv8s / YOLO11s (about 20 MB) or YOLOv8m (about 50 MB), and 960 px resolution, for better results at the cost of speed.
* **Auto mode (about 2,000 names) is an estimate.** The wide-vocabulary detector can mislabel unusual scenes. Out-of-focus blobs are filtered and names below a 0.40 score are hidden, but errors can still happen. Raise the Confidence slider for stricter results.
* **Vehicle types and models** are CLIP zero-shot guesses, not a trained car-model recogniser. Types (hatchback, scooter, tempo) are usually right. Exact models (for example Swift or Splendor) are often wrong on small or blurry vehicles. They are available for images and video files only.
* **Tracking:** SORT, ByteTrack and BoT-SORT (no Deep SORT). IDs can switch when objects are hidden behind each other for a long time.
* **Live webcam** works in the browser at localhost. **RTSP / IP camera is marked "coming soon"** in the website and is not part of the tested feature set (the code exists in `backend/streams.py`).
* **Traffic Survey Report** is available for uploaded videos only.
* **Videos** are processed once, completely. Long videos take proportionally longer, and CPU-only machines are slow with the larger models.

**Free upgrades planned:** larger free detectors (YOLOv8l/x, RT-DETR), Deep SORT or BoT-SORT appearance tracking, a free local vision-language model to review doubtful detections, a fine-tuned vehicle-model classifier on a public dataset, and live-mode vehicle recognition.

**Licence note:** Ultralytics YOLO is AGPL-3.0. It is free for study, research and portfolio projects. A commercial product needs an Ultralytics licence.

---

## About

Built by **[Your Name]** for the **CodeAlpha** internship.

* GitHub: [add link]
* LinkedIn: [add link]
* Demo videos: [add links]
