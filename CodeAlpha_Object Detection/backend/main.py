"""
The Royal Court of Vision - FastAPI backend (and static host for the website).

  POST /api/jobs                 upload a video (multipart)        -> {"id"}
  GET  /api/jobs/{id}            poll status / progress / stats
  GET  /api/jobs/{id}/video      tracked MP4  (add ?download=1 to download)
  GET  /api/jobs/{id}/heatmap    heatmap PNG
  GET  /api/jobs/{id}/csv        full track log
  GET  /api/jobs/{id}/rows       first 30 ledger rows (JSON)
  POST /api/image                detect objects in one image       -> result JSON
  GET  /api/jobs/{id}/image      annotated image
  WS   /ws/live                  webcam frames in, tracks out
  POST /api/stream/rtsp-connect  open an RTSP / IP-camera stream
  GET  /api/stream/{sid}/mjpeg   annotated MJPEG stream
  GET  /api/stream/{sid}/stats   live counters
  DELETE /api/stream/{sid}       stop a stream
  GET  /api/health               status check
Interactive docs: /docs
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine
import streams
from vocab import COCO_SET

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.environ.get("RCV_FRONTEND_DIR", ROOT.parent / "frontend"))
WORK_DIR = Path(os.environ.get("RCV_WORK_DIR", Path(tempfile.gettempdir()) / "rcv_jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.environ.get("RCV_MAX_UPLOAD_MB", "120"))
JOB_TTL_S = int(os.environ.get("RCV_JOB_TTL_S", "3600"))
VIDEO_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="The Royal Court of Vision API", version=VERSION,
              description="YOLOv8 detection + SORT / ByteTrack / BoT-SORT tracking with line counting, heatmaps and CSV logs.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["Content-Disposition", "Content-Range", "Accept-Ranges"])

JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("RCV_WORKERS", "1")))
LIVE_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ───────────────────────── helpers ─────────────────────────
def _clean_params(tracker: str, model: str, conf: float, classes: str, line: int, max_frames: int = 300, imgsz: int = 640,
                  custom: str = "", tiles: bool = False, people: bool = False, auto: bool = False, vehicles: bool = True,
                  report_bin: int = 0, expected_dir: str = "", stop_seconds: float = 10.0) -> dict:
    tracker = (tracker or "bytetrack").lower()
    if tracker not in engine.ALLOWED_TRACKERS:
        raise HTTPException(400, f"tracker must be one of {sorted(engine.ALLOWED_TRACKERS)}")
    if model not in engine.ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {sorted(engine.ALLOWED_MODELS)}")
    if int(imgsz) not in engine.ALLOWED_IMGSZ:
        raise HTTPException(400, f"imgsz must be one of {list(engine.ALLOWED_IMGSZ)}")
    cls = [c.strip().lower() for c in (classes or "").split(",") if c.strip()]
    if not cls and not auto:
        cls = ["person", "car"]
    extra = []
    for w in (custom or "").replace("\n", ",").split(","):
        w = " ".join(w.strip().lower().split())[:40]
        if w and w not in extra:
            extra.append(w)
    extra = extra[:80]
    outside = [c for c in cls if c not in COCO_SET]
    vocab = list(dict.fromkeys(cls + extra)) if (extra or outside) and not auto else None
    if auto:  # Auto = every standard class for video / live / RTSP (instant); images use the wide vocabulary instead
        cls = [] if not extra else cls
        vocab = list(dict.fromkeys(extra)) if extra else None
    if vocab:  # open-vocabulary names need the YOLO-World model
        model = model if "world" in model else engine.WORLD_DEFAULT
    return {"tracker": tracker, "model": model, "conf": min(0.95, max(0.05, float(conf))),
            "classes": cls, "line": int(min(95, max(5, line))), "max_frames": int(max(0, max_frames)),   # 0 = the whole video
            "imgsz": int(imgsz), "vocab": vocab, "tiles": bool(tiles), "people": bool(people),
            "auto": bool(auto), "extra": extra, "vehicles": bool(vehicles),
            "report_bin": int(max(0, min(report_bin, 3600))), "expected_dir": expected_dir if expected_dir in ("up", "down", "left", "right") else "",
            "stop_seconds": float(max(2, min(stop_seconds, 600)))}


async def _save_upload(file: UploadFile, dest: Path) -> int:
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            f.write(chunk)
    if size == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    return size


def _new_job(kind: str, params: dict) -> tuple[str, dict, Path]:
    jid = uuid.uuid4().hex[:10]
    d = WORK_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    job = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "error": None, "created": time.time(),
           "stats": {}, "ledger": [], **params}
    JOBS[jid] = job
    return jid, job, d


def _get_job(jid: str) -> dict:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Job not found. It may have expired - please run it again.")
    return job


def _job_file(jid: str, name: str, media: str, download_as: Optional[str] = None, download: bool = False):
    job = _get_job(jid)
    if job["status"] != "done":
        raise HTTPException(409, "Job is not finished yet.")
    p = WORK_DIR / jid / name
    if not p.exists():
        raise HTTPException(404, "File not available for this job.")
    headers = {"Content-Disposition": f'attachment; filename="{download_as}"'} if (download and download_as) else None
    return FileResponse(p, media_type=media, headers=headers)


def _cleanup_loop():
    while True:
        time.sleep(300)
        now = time.time()
        for jid, job in list(JOBS.items()):
            if now - job["created"] > JOB_TTL_S and job["status"] in ("done", "error"):
                JOBS.pop(jid, None)
                shutil.rmtree(WORK_DIR / jid, ignore_errors=True)


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ───────────────────────── video jobs ─────────────────────────
def _worker(jid: str, in_path: Path):
    job = JOBS[jid]
    job["status"] = "running"
    try:
        engine.run_video_job(job, in_path, WORK_DIR / jid, lambda p: job.__setitem__("progress", round(p, 3)))
        job["progress"], job["status"] = 1.0, "done"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e) or e.__class__.__name__
    finally:
        in_path.unlink(missing_ok=True)


@app.post("/api/jobs", tags=["video"])
async def create_job(file: UploadFile = File(...), tracker: str = Form("bytetrack"), model: str = Form("yolov8n.pt"),
                     conf: float = Form(0.35), classes: str = Form("person,car"), line: int = Form(50),
                     max_frames: int = Form(0), imgsz: int = Form(640), custom: str = Form(""),
                     people: bool = Form(False), auto: bool = Form(False), vehicles: bool = Form(True),
                     report_bin: int = Form(0), expected_dir: str = Form(""), stop_seconds: float = Form(10.0)):
    """Upload a video; processing runs in the background - poll GET /api/jobs/{id}."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, "Please upload a video file (MP4, MOV, WEBM, AVI or MKV).")
    if not auto and not (classes or "").strip() and not (custom or "").strip():
        raise HTTPException(400, "Pick at least one target in the menus, or switch to Auto to detect everything.")
    params = _clean_params(tracker, model, conf, classes, line, max_frames, imgsz, custom, False, people, auto, vehicles, report_bin, expected_dir, stop_seconds)
    jid, job, d = _new_job("video", params)
    in_path = d / f"input{ext}"
    try:
        await _save_upload(file, in_path)
    except HTTPException:
        JOBS.pop(jid, None)
        shutil.rmtree(d, ignore_errors=True)
        raise
    EXECUTOR.submit(_worker, jid, in_path)
    return {"id": jid}


@app.get("/api/jobs/{jid}", tags=["video"])
def job_status(jid: str):
    j = _get_job(jid)
    return {"id": jid, "kind": j["kind"], "status": j["status"], "progress": j["progress"], "error": j["error"],
            "stats": j["stats"], "width": j.get("width"), "height": j.get("height")}


@app.get("/api/jobs/{jid}/video", tags=["video"])
def job_video(jid: str, download: int = 0):
    return _job_file(jid, "tracked.mp4", "video/mp4", f"royal-court-tracked-{jid}.mp4", bool(download))


@app.get("/api/jobs/{jid}/heatmap", tags=["video"])
def job_heatmap(jid: str, download: int = 0):
    return _job_file(jid, "heatmap.png", "image/png", f"royal-court-heatmap-{jid}.png", bool(download))


@app.get("/api/jobs/{jid}/csv", tags=["video"])
def job_csv(jid: str):
    return _job_file(jid, "tracks.csv", "text/csv", f"royal-court-tracks-{jid}.csv", True)


@app.get("/api/jobs/{jid}/report", tags=["video"])
def job_report(jid: str, download: int = 0):
    """Traffic Survey Report (a self-contained page: open it, or print it to PDF)."""
    return _job_file(jid, "report.html", "text/html", f"traffic-survey-report-{jid}.html", bool(download))


@app.get("/api/jobs/{jid}/report.csv", tags=["video"])
def job_report_csv(jid: str):
    return _job_file(jid, "report.csv", "text/csv", f"traffic-survey-report-{jid}.csv", True)


@app.get("/api/jobs/{jid}/rows", tags=["video"])
def job_rows(jid: str):
    return {"rows": _get_job(jid)["ledger"]}


# ───────────────────────── image mode ─────────────────────────
@app.post("/api/image", tags=["image"])
async def detect_image(file: UploadFile = File(...), model: str = Form("yolov8n.pt"), conf: float = Form(0.35),
                       classes: str = Form("person,car"), tracker: str = Form("bytetrack"), line: int = Form(50),
                       imgsz: int = Form(960), custom: str = Form(""), tiles: bool = Form(False),
                       people: bool = Form(False), auto: bool = Form(True), vehicles: bool = Form(True)):
    """Detect objects in a single image (tracking needs video, so only detection runs here)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXT:
        raise HTTPException(400, "Please upload an image (JPG, PNG, WEBP or BMP).")
    if not auto and not (classes or "").strip() and not (custom or "").strip():
        raise HTTPException(400, "Pick at least one target in the menus, or switch to Auto to detect everything.")
    params = _clean_params(tracker, model, conf, classes, line, 300, imgsz, custom, tiles, people, auto, vehicles)
    jid, job, d = _new_job("image", params)
    in_path = d / f"input{ext}"
    try:
        await _save_upload(file, in_path)
        job["status"] = "running"
        await asyncio.get_running_loop().run_in_executor(LIVE_EXECUTOR, engine.run_image_job, job, in_path, d)
        job["status"], job["progress"] = "done", 1.0
    except HTTPException:
        JOBS.pop(jid, None)
        shutil.rmtree(d, ignore_errors=True)
        raise
    except engine.ModelPreparing as e:
        job["status"], job["error"] = "error", str(e)
        raise HTTPException(503, job["error"])
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e) or e.__class__.__name__
        raise HTTPException(422, job["error"])
    finally:
        in_path.unlink(missing_ok=True)
    return {"id": jid, "kind": "image", "status": "done", "progress": 1.0, "stats": job["stats"],
            "width": job["width"], "height": job["height"]}


@app.get("/api/jobs/{jid}/image", tags=["image"])
def job_image(jid: str, download: int = 0):
    return _job_file(jid, "annotated.jpg", "image/jpeg", f"royal-court-detected-{jid}.jpg", bool(download))


# ───────────────────────── webcam over WebSocket ─────────────────────────
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Browser sends JPEG frames (binary); server answers with JSON tracks per frame (coordinates normalised 0-1)."""
    await ws.accept()
    q = ws.query_params
    try:
        params = _clean_params(q.get("tracker", "bytetrack"), q.get("model", "yolov8n.pt"), float(q.get("conf", 0.35)),
                               q.get("classes", "person,car"), int(q.get("line", 50)), 300, int(q.get("imgsz", 640)),
                               q.get("custom", ""), False, q.get("people") == "1", q.get("auto") == "1")
        loop = asyncio.get_running_loop()
        sess = await loop.run_in_executor(LIVE_EXECUTOR, lambda: engine.TrackingSession(
            params["model"], params["tracker"], params["conf"], params["classes"], params["line"], params["imgsz"], params["vocab"], params["people"]))
    except Exception as e:  # noqa: BLE001
        await ws.send_text(json.dumps({"error": str(e)}))
        await ws.close()
        return

    def work(data: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "bad frame"}
        h, w = frame.shape[:2]
        tracks = sess.step(frame)
        st = sess.stats()
        return {"tracks": [{"id": t["id"], "cls": t["cls"], "conf": round(t["conf"], 2),
                            "box": [t["box"][0] / w, t["box"][1] / h, t["box"][2] / w, t["box"][3] / h],
                            "crossed": t["crossed"], "direction": t["direction"], "attrs": t.get("attrs", "")} for t in tracks],
                "down": st["down"], "up": st["up"], "unique": st["unique"], "frames": st["frames"],
                "fps": st["fps"], "per_class": st["per_class"], "max_concurrency": st["max_concurrency"],
                "people_summary": st.get("people_summary", {})}

    try:
        while True:
            data = await ws.receive_bytes()
            result = await loop.run_in_executor(LIVE_EXECUTOR, work, data)
            await ws.send_text(json.dumps(result))
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:  # noqa: BLE001
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ───────────────────────── RTSP / IP camera ─────────────────────────
class RtspRequest(BaseModel):
    url: str
    transport: str = "tcp"
    latency: int = 200
    tracker: str = "bytetrack"
    model: str = "yolov8n.pt"
    conf: float = 0.35
    classes: str = "person,car"
    line: int = 50
    imgsz: int = 640
    custom: str = ""
    people: bool = False
    auto: bool = False


@app.post("/api/stream/rtsp-connect", tags=["stream"])
async def rtsp_connect(req: RtspRequest):
    u = urlparse(req.url.strip())
    if u.scheme.lower() not in ("rtsp", "rtsps", "http", "https") or not u.hostname:
        raise HTTPException(400, "Stream URL must start with rtsp:// (or http:// for an MJPEG IP camera).")
    p = _clean_params(req.tracker, req.model, req.conf, req.classes, req.line, 300, req.imgsz, req.custom, False, req.people, req.auto)
    params = {"model": p["model"], "tracker": p["tracker"], "conf": p["conf"], "classes": p["classes"], "line_pct": p["line"], "imgsz": p["imgsz"],
              "vocab": p["vocab"], "people": p["people"]}
    try:
        s = await asyncio.get_running_loop().run_in_executor(None, streams.create_session, req.url.strip(), req.transport, params)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    except ConnectionError as e:
        raise HTTPException(502, str(e))
    return {"status": "connected", "sessionId": s.id, "url": req.url, "transport": req.transport,
            "mjpeg": f"/api/stream/{s.id}/mjpeg", "message": "Stream opened. Tracking is running on the server."}


def _stream(sid: str) -> streams.StreamSession:
    s = streams.SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Stream session not found or already stopped.")
    return s


@app.get("/api/stream/{sid}/mjpeg", tags=["stream"])
async def stream_mjpeg(sid: str):
    s = _stream(sid)

    async def gen():
        last = None
        while not s.stop_flag.is_set():
            s.last_seen = time.time()
            if s.latest is not None and s.latest is not last:
                last = s.latest
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(last)).encode() + b"\r\n\r\n" + last + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/stream/{sid}/stats", tags=["stream"])
def stream_stats(sid: str):
    s = _stream(sid)
    s.last_seen = time.time()
    return {"running": not s.stop_flag.is_set(), "error": s.error, **s.stats}


@app.delete("/api/stream/{sid}", tags=["stream"])
def stream_stop(sid: str):
    s = streams.SESSIONS.get(sid)
    if s:
        s.stop()
    return {"status": "stopped"}


# ───────────────────────── health + website ─────────────────────────
@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "version": VERSION, "detector": "fake" if engine.FAKE else "yolov8",
            "models": sorted(engine.ALLOWED_MODELS), "imgsz": list(engine.ALLOWED_IMGSZ), "trackers": sorted(engine.ALLOWED_TRACKERS),
            "max_upload_mb": MAX_UPLOAD_MB, "warmup": engine.warmup_status()}


@app.on_event("startup")
def _prepare_models_in_background():
    """Download / bake the models once, in the background, so no visitor ever waits for it."""
    if os.environ.get("RCV_NO_WARMUP") != "1" and not engine.FAKE:
        engine.start_warmup(people=os.environ.get("RCV_PRELOAD_PEOPLE", "1") != "0")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="site")
else:
    @app.get("/")
    def root():
        return JSONResponse({"name": "The Royal Court of Vision API", "docs": "/docs", "health": "/api/health"})
