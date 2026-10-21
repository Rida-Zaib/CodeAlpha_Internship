"""
Royal Court of Vision - detection + tracking engine.

Pipeline for every frame:
    OpenCV frame  ->  YOLOv8 detection  ->  tracker (SORT / ByteTrack / BoT-SORT)
                  ->  line-crossing counter  ->  annotated frame (boxes, IDs, trails)

The same TrackingSession class powers all four inputs of the website:
video upload, image upload, webcam (WebSocket) and RTSP / IP camera streams.
"""
from __future__ import annotations

import contextlib
import csv
import os
import shutil
import subprocess
import threading
import time
import types
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from attributes import get_analyzer, person_crop, summarize
from sort_tracker import Sort, iou_batch
from vocab import CATEGORIES, COCO_80, COCO_SET, DETECT_SET, DETECT_VOCAB, UNIVERSAL_VOCAB
import verify
import report as traffic_report
import vehicles

# ───────────────────────── configuration ─────────────────────────
ALLOWED_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt", "yolov8s-worldv2.pt", "yolov8m-worldv2.pt"}
WORLD_DEFAULT = "yolov8s-worldv2.pt"  # open-vocabulary detector used when custom object names are given
ALLOWED_IMGSZ = (480, 640, 960, 1280)
ALLOWED_TRACKERS = {"sort", "bytetrack", "botsort"}
MODELS_DIR = Path(os.environ.get("RCV_MODELS_DIR", Path(__file__).resolve().parent / "models"))
FAKE = os.environ.get("RCV_FAKE_DETECTOR") == "1"  # synthetic detector, used by the self-test
MAX_WIDTH = int(os.environ.get("RCV_MAX_WIDTH", "960"))  # frames wider than this are downscaled (speed)
INFER_LOCK = threading.Lock()  # one neural-net forward pass at a time (keeps a free CPU server stable)

# Site palette (BGR): sage-cream, gold, mist teal, fog, dusk teal
PALETTE = [(184, 230, 223), (138, 205, 227), (160, 154, 111), (187, 184, 157), (200, 214, 150), (120, 190, 215)]
DEEP_TEAL = (58, 48, 6)
GOLD = (138, 205, 227)
SAGE = (184, 230, 223)

_MODEL_CACHE: dict = {}


# ───────────────────────── model loading ─────────────────────────
ROOT = Path(__file__).resolve().parent
UNIVERSAL_FILE = f"universal-world-{len(DETECT_VOCAB)}.pt"   # YOLO-World with the big vocabulary baked in (re-baked automatically when the vocabulary changes)
CLIP_FILE = ROOT / "weights" / "clip" / "ViT-B-32.pt"
CLIP_BYTES = 353_976_522                # size of the CLIP file, used for the progress bar
_CHDIR_LOCK = threading.RLock()


@contextlib.contextmanager
def _in_dir(path: Path):
    """Run a block with a fixed working directory. Ultralytics and CLIP download into the *current* folder,
    so without this a model is downloaded again whenever the server is started from a different folder."""
    with _CHDIR_LOCK:
        path.mkdir(parents=True, exist_ok=True)
        old = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)


def clip_ready() -> bool:
    try:
        return CLIP_FILE.exists() and CLIP_FILE.stat().st_size >= CLIP_BYTES * 0.99
    except OSError:
        return False


def clip_progress() -> float:
    try:
        return min(1.0, CLIP_FILE.stat().st_size / CLIP_BYTES) if CLIP_FILE.exists() else 0.0
    except OSError:
        return 0.0


def universal_ready() -> bool:
    return (MODELS_DIR / UNIVERSAL_FILE).exists()


def _load_yolo(name: str):
    from ultralytics import YOLO  # imported lazily: keeps start-up and tests light

    local = MODELS_DIR / name
    if local.exists():
        return YOLO(str(local))
    with _in_dir(MODELS_DIR):  # a first download lands in backend/models, whatever folder the server was started from
        return YOLO(name)


def _load_world(name: str):
    from ultralytics import YOLOWorld

    local = MODELS_DIR / name
    if local.exists():
        return YOLOWorld(str(local))
    with _in_dir(MODELS_DIR):
        return YOLOWorld(name)


class ModelPreparing(RuntimeError):
    """Raised when a feature needs a one-time download that is still running in the background."""


def _load_model(name: str, vocab=None):
    """Standard YOLO, or YOLO-World (open vocabulary) when `vocab` is given."""
    if vocab:
        if not clip_ready():
            if WARMUP["state"] == "running":
                raise ModelPreparing("The wide-vocabulary model is still being prepared (first run only, "
                                     f"{int(clip_progress() * 100)}%). Try again in a moment - or clear the custom names.")
            if not os.environ.get("RCV_ALLOW_BLOCKING_DOWNLOAD"):
                start_warmup()
                raise ModelPreparing("The wide-vocabulary model has started preparing in the background (one-time download). "
                                     "Try again in a few minutes - or clear the custom names.")
        m = _load_world(name)
        try:
            with _in_dir(ROOT):
                m.set_classes(list(vocab))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Open-vocabulary mode could not start (" + str(e)[:160] + "). It needs the CLIP text encoder - try: "
                "pip install git+https://github.com/ultralytics/CLIP.git  (requires git), or clear the custom objects box.") from e
        return m
    return _load_yolo(name)


def get_shared_model(name: str, vocab=None):
    """Cached model for stateless prediction (SORT / image mode)."""
    key = (name, tuple(vocab or ()))
    if key not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= 4:  # keep memory small: drop the oldest
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = _load_model(name, vocab)
    return _MODEL_CACHE[key]


_UNIVERSAL = None


def get_universal_model():
    """YOLO-World with the whole vocabulary baked in. Loading it needs no CLIP and no download."""
    global _UNIVERSAL
    if _UNIVERSAL is None:
        _UNIVERSAL = _load_world(UNIVERSAL_FILE)
    return _UNIVERSAL


# ───────────────────────── one-time preparation (runs in the background) ─────────────────────────
WARMUP: dict = {"state": "idle", "step": "", "error": None, "people": False}
_WARM_LOCK = threading.Lock()


def build_universal_model() -> None:
    """Download YOLO-World + CLIP once, bake UNIVERSAL_VOCAB into the weights and save them."""
    target = MODELS_DIR / UNIVERSAL_FILE
    if target.exists():
        return
    m = _load_world(WORLD_DEFAULT)
    with _in_dir(ROOT):  # CLIP lands in backend/weights/clip, the same place every time
        m.set_classes(list(DETECT_VOCAB))
    tmp = MODELS_DIR / (UNIVERSAL_FILE + ".tmp")
    m.save(str(tmp))
    os.replace(tmp, target)


def warmup(people: bool = False) -> None:
    try:
        WARMUP.update(state="running", error=None)
        WARMUP["step"] = "Standard detector"
        for n in ("yolov8n.pt", "yolov8s.pt"):
            if not FAKE and not (MODELS_DIR / n).exists():
                _load_yolo(n)
        if not FAKE:
            WARMUP["step"] = "Wide-vocabulary model (one-time download)"
            build_universal_model()
        if people:
            WARMUP["step"] = "People analysis model"
            get_analyzer().ensure_loaded()
        WARMUP.update(state="ready", step="")
    except Exception as e:  # noqa: BLE001
        WARMUP.update(state="error", error=str(e)[:200] or e.__class__.__name__)


def start_warmup(people: bool = False) -> None:
    """Start the preparation in a background thread (safe to call many times)."""
    with _WARM_LOCK:
        if WARMUP["state"] == "running" or FAKE:
            return
        WARMUP["state"] = "running"
        threading.Thread(target=warmup, args=(people,), daemon=True, name="rcv-warmup").start()


def warmup_status() -> dict:
    return {"state": WARMUP["state"], "step": WARMUP["step"], "error": WARMUP["error"],
            "universal_ready": universal_ready() or FAKE, "clip_progress": round(clip_progress(), 3),
            "clip_ready": clip_ready()}


COCO_NAMES = {}  # filled on first model load; fake detector uses its own subset


def class_ids_for(names: dict, wanted: list[str]) -> Optional[list[int]]:
    wanted = [w.strip().lower() for w in wanted if w.strip()]
    ids = [i for i, n in names.items() if n.lower() in wanted]
    return ids or None  # None = every class



# ───────────────────────── duplicate / false-detection control ─────────────────────────
_PERSONS = {"person", "man", "woman", "boy", "girl", "child", "baby", "toddler", "teenager", "pedestrian", "passenger", "driver",
            "rider", "cyclist", "motorcyclist", "student", "worker", "police officer", "soldier"}
_FAMILY: dict[str, str] = {}
for _n in _PERSONS:
    _FAMILY[_n] = "person"
for _g, _fam in (("car", "motor"), ("truck", "motor"), ("bus", "motor"), ("motorcycle", "twowheel"), ("bicycle", "twowheel")):
    _FAMILY[_g] = _fam
    for _t in vehicles.TYPES[_g]:
        _FAMILY.setdefault(_t, _fam)
for _k, _t, _items in CATEGORIES:
    if _k == "animals":
        for _n in _items:
            _FAMILY.setdefault(_n, "animal")
for _n in ("van", "taxi", "jeep", "ambulance", "police car", "pickup truck", "tractor", "auto rickshaw", "e-rickshaw", "tempo"):
    _FAMILY.setdefault(_n, "motor")


def family_of(name: str) -> str:
    """Objects of one family are alternative names for the same thing (car / truck / van ...): only one may survive on a spot."""
    return _FAMILY.get(name, name)


def _iou_contain(a, b) -> tuple[float, float]:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa, ab = max(1e-9, (a[2] - a[0]) * (a[3] - a[1])), max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + ab - inter), inter / min(aa, ab)


def same_object(a, b, iou: float = 0.6, contain: float = 0.9) -> bool:
    i, c = _iou_contain(a, b)
    return i >= iou or c >= contain


def dedupe_tracks(tracks: list[dict], iou: float = 0.6, contain: float = 0.9) -> list[dict]:
    """One box per object: of two same-family boxes on the same spot (truck + car on one car) the surer one survives."""
    keep: list[dict] = []
    for t in sorted(tracks, key=lambda t: -t["conf"]):
        f = family_of(t["cls"])
        if not any(family_of(k["cls"]) == f and same_object(k["box"], t["box"], iou, contain) for k in keep):
            keep.append(t)
    return keep


_CROWDED = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "bird", "sheep", "cow"}


def filter_dets(dets: np.ndarray, names: dict, W: int, H: int, tiled: bool = False) -> np.ndarray:
    """Image mode: drop specks and same-family duplicates (rows are x1,y1,x2,y2,conf,cls)."""
    if not len(dets):
        return dets
    bw, bh = dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1]
    ok = (bw >= 10) & (bh >= 10) & (bw * bh >= (0.00008 if tiled else 0.0002) * W * H)
    dets = dets[ok]
    if not len(dets):
        return dets
    order = np.argsort(-dets[:, 4])
    keep: list[int] = []
    for i in order:
        fi = family_of(names.get(int(dets[i, 5]), str(int(dets[i, 5]))))
        nm = names.get(int(dets[i, 5]), "")
        loose = nm not in _CROWDED     # animals / things: two boxes of one name that overlap a lot are one object (a stump named "squirrel")
        if not any(family_of(names.get(int(dets[k, 5]), "")) == fi and
                   (same_object(dets[k, :4], dets[i, :4]) or (loose and names.get(int(dets[k, 5]), "") == nm and
                                                               _iou_contain(dets[k, :4], dets[i, :4])[1] >= 0.25))
                   for k in keep):
            keep.append(int(i))
    return dets[sorted(keep, key=lambda k: -dets[k, 4])]


def consolidate(raw: list[np.ndarray], names: dict, fps: float, conf: float) -> tuple[list[list[dict]], dict]:
    """
    Second pass over everything the detector saw in the WHOLE video:
      1. drop flicker  - tracks that live for under ~0.25 s or never reach a clearly confident frame,
      2. merge doubles - two tracks of the same family sitting on the same box for most of their life are one object,
      3. one stable class per object - confidence-weighted vote over its whole life (no frame-to-frame flipping),
      4. clean ids     - 1, 2, 3 ... in order of appearance.
    Returns (per-frame track dicts, info).
    """
    n_f = len(raw)
    st: dict[int, dict] = {}
    for f, rows in enumerate(raw):
        for x1, y1, x2, y2, tid, cf, cid in rows:
            tid = int(tid)
            s = st.setdefault(tid, {"n": 0, "max": 0.0, "votes": {}, "first": f})
            s["n"] += 1
            s["max"] = max(s["max"], float(cf))
            c = names.get(int(cid), str(int(cid)))
            s["votes"][c] = s["votes"].get(c, 0.0) + float(cf)
    min_len = max(2, min(int(round(fps * 0.25)), max(1, n_f // 3)))
    ok = {t for t, s in st.items() if s["n"] >= min_len and (s["max"] >= conf + 0.10 or s["n"] >= 4 * min_len)}
    removed_short = len(st) - len(ok)

    # --- merge duplicate tracks (union-find) ---
    parent = {t: t for t in ok}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    over: dict[tuple[int, int], int] = {}
    for rows in raw:
        cur = [(int(r[4]), r[:4], names.get(int(r[6]), str(int(r[6])))) for r in rows if int(r[4]) in ok]
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if cur[i][0] != cur[j][0] and family_of(cur[i][2]) == family_of(cur[j][2]) and same_object(cur[i][1], cur[j][1]):
                    k = tuple(sorted((cur[i][0], cur[j][0])))
                    over[k] = over.get(k, 0) + 1
    merged = 0
    for (a, b), c in sorted(over.items(), key=lambda kv: -kv[1]):
        ra, rb = find(a), find(b)
        if ra == rb or c < 0.5 * min(st[ra]["n"], st[rb]["n"]):
            continue
        big, small = (ra, rb) if st[ra]["n"] >= st[rb]["n"] else (rb, ra)
        parent[small] = big
        st[big]["n"] += st[small]["n"]
        for k, v in st[small]["votes"].items():
            st[big]["votes"][k] = st[big]["votes"].get(k, 0.0) + v
        st[big]["first"] = min(st[big]["first"], st[small]["first"])
        merged += 1

    # --- per-frame rows, one box per object, then clean ids ---
    # mild prior: when a detector hesitates between car and truck on the same box, cars are the more common answer
    prior = {"car": 1.25}
    final_cls = {r: max(st[r]["votes"].items(), key=lambda kv: kv[1] * prior.get(kv[0], 1.0))[0] for r in {find(t) for t in ok}}
    order = sorted(final_cls, key=lambda r: (st[r]["first"], r))
    new_id = {r: i + 1 for i, r in enumerate(order)}
    frames: list[list[dict]] = []
    for rows in raw:
        best: dict[int, dict] = {}
        for x1, y1, x2, y2, tid, cf, cid in rows:
            tid = int(tid)
            if tid not in ok:
                continue
            r = find(tid)
            if r not in best or cf > best[r]["conf"]:
                best[r] = {"id": new_id[r], "cls": final_cls[r], "conf": float(cf), "box": [float(x1), float(y1), float(x2), float(y2)], "_root": r}
        frames.append(dedupe_tracks(list(best.values()), 0.7, 0.92))
    return frames, {"raw_tracks": len(st), "removed_short": removed_short, "merged": merged, "kept": len(final_cls),
                    "min_frames": min_len, "raw_to_new": {t: new_id[find(t)] for t in ok}, "frames": n_f}


# ───────────────────────── detector + tracker ─────────────────────────
class Detector:
    """YOLOv8 detection plus the chosen tracker, returning [x1,y1,x2,y2,id,conf,cls] rows."""

    FAKE_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog", 24: "backpack"}

    def __init__(self, model: str, tracker: str, conf: float, classes: list[str], imgsz: int = 640, vocab=None):
        self.imgsz = imgsz
        self.kind = tracker
        self.conf = conf
        self.model_name = model
        self.frame_no = 0
        if FAKE:
            self.names = self.FAKE_NAMES
            self.model = None
        else:
            # ByteTrack / BoT-SORT keep state inside the model, so they need their own instance.
            self.model = get_shared_model(model, vocab) if tracker == "sort" else _load_model(model, vocab)
            self.names = self.model.names
        self.class_ids = None if vocab else class_ids_for(self.names, classes)
        self.sort = Sort(max_age=30, min_hits=2, iou_threshold=0.3) if (tracker == "sort" or FAKE) else None

    # --- synthetic detections for tests only ---
    def _fake(self, frame):
        h, w = frame.shape[:2]
        t = self.frame_no
        dets = []
        for k, cls in enumerate([0, 0, 2, 2, 1]):
            bw, bh = (w * 0.07, h * 0.28) if cls == 0 else (w * 0.16, h * 0.18)
            cx = (w * (0.15 + 0.17 * k) + t * (3 + k)) % w
            cy = (h * 0.2 + (t * (2 + k) + k * 90) % (h * 0.7)) if k % 2 == 0 else (h * 0.9 - (t * (2 + k) + k * 60) % (h * 0.7))
            dets.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 0.8 + 0.02 * k, cls])
        dets = np.array(dets, dtype=float)
        if self.class_ids:
            dets = dets[np.isin(dets[:, 5], self.class_ids)]
        return dets

    def track(self, frame: np.ndarray) -> np.ndarray:
        self.frame_no += 1
        if FAKE:
            return self.sort.update(self._fake(frame))
        if self.kind == "sort":
            with INFER_LOCK:
                r = self.model.predict(frame, conf=self.conf, classes=self.class_ids, imgsz=self.imgsz, agnostic_nms=True, verbose=False)[0]
            b = r.boxes
            if b is None or len(b) == 0:
                return self.sort.update(np.zeros((0, 6)))
            dets = np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]])
            return self.sort.update(dets)
        # ByteTrack / BoT-SORT are built into Ultralytics
        yaml = "bytetrack.yaml" if self.kind == "bytetrack" else "botsort.yaml"
        with INFER_LOCK:
            r = self.model.track(frame, persist=True, tracker=yaml, conf=self.conf, classes=self.class_ids,
                                 imgsz=self.imgsz, agnostic_nms=True, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0 or b.id is None:
            return np.zeros((0, 7))
        return np.hstack([b.xyxy.cpu().numpy(), b.id.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None],
                          b.cls.cpu().numpy()[:, None]])


# ───────────────────────── tracking session (counting + drawing) ─────────────────────────
class TrackingSession:
    def __init__(self, model="yolov8n.pt", tracker="bytetrack", conf=0.35, classes=("person", "car"), line_pct=50, imgsz=640, vocab=None, people=False,
                 render_only=False):
        # render_only: no detector - used to draw / count tracks that were already found and cleaned up
        self.det = types.SimpleNamespace(kind=tracker, names={}) if render_only else Detector(model, tracker, conf, list(classes), imgsz, vocab)
        self.details: dict[int, str] = {}   # track id -> vehicle type / model text
        self.analyzer = None
        if people:
            a = get_analyzer()
            if a.is_ready():
                self.analyzer = a
            else:
                a.load_async()  # downloads in the background; this session runs without person details
        self.attrs: dict[int, dict] = {}
        self.names = self.det.names
        self.line_pct = line_pct
        self.trails: dict[int, deque] = {}
        self.last_cy: dict[int, float] = {}
        self.counted: set = set()
        self.id_class: dict[int, str] = {}
        self.down = 0
        self.up = 0
        self.max_conc = 0
        self.frames = 0
        self.flash = 0
        self.infer_ms = deque(maxlen=30)

    # --- one frame in, list of track dicts out ---
    def step(self, frame: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        rows = self.det.track(frame)
        self.infer_ms.append((time.perf_counter() - t0) * 1000)
        return self.process(rows, frame, live=True)

    def process(self, rows: np.ndarray, frame: np.ndarray, live: bool = False) -> list[dict]:
        """Counting, trails and person details for one frame of [x1,y1,x2,y2,id,conf,cls] rows."""
        h, w = frame.shape[:2]
        line_y = h * self.line_pct / 100.0
        self.frames += 1
        if live and len(rows) > 1:  # one box per object, before anything is counted
            cand = [{"cls": self.names.get(int(r[6]), str(int(r[6]))), "conf": float(r[5]), "box": list(r[:4]), "i": k} for k, r in enumerate(rows)]
            keepi = {t["i"] for t in dedupe_tracks(cand)}
            rows = [r for k, r in enumerate(rows) if k in keepi]
        out = []
        for x1, y1, x2, y2, tid, conf, cid in rows:
            tid, cid = int(tid), int(cid)
            cls = self.names.get(cid, str(cid))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.id_class[tid] = cls
            direction = None
            prev = self.last_cy.get(tid)
            if prev is not None and (prev - line_y) * (cy - line_y) < 0:
                direction = "down" if cy > prev else "up"
                if (tid, direction) not in self.counted:
                    self.counted.add((tid, direction))
                    if direction == "down":
                        self.down += 1
                    else:
                        self.up += 1
                    self.flash = 8
                else:
                    direction = None
            self.last_cy[tid] = cy
            self.trails.setdefault(tid, deque(maxlen=40)).append((int(cx), int(cy)))
            out.append({"id": tid, "cls": cls, "conf": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)],
                        "cx": float(cx), "cy": float(cy), "crossed": direction is not None, "direction": direction})
        for t in out:
            if t["id"] in self.details:
                t["attrs"] = self.details[t["id"]]
        self.max_conc = max(self.max_conc, len(out))
        if self.analyzer is not None:  # attributes once per new person track (max 4 per frame keeps it real-time)
            todo = [t for t in out if t["cls"] == "person" and t["id"] not in self.attrs][:4]
            crops = [(t, person_crop(frame, t["box"])) for t in todo]
            crops = [(t, c) for t, c in crops if c is not None]
            if crops:
                for (t, _), a in zip(crops, self.analyzer.analyze([c for _, c in crops])):
                    self.attrs[t["id"]] = a
            for t in out:
                if t["id"] in self.attrs:
                    t["attrs"] = self.attrs[t["id"]]["tag"]
        return out

    # --- numbers for the dashboard ---
    def per_class(self) -> dict:
        d: dict = {}
        for c in self.id_class.values():
            d[c] = d.get(c, 0) + 1
        return d

    def stats(self) -> dict:
        ms = sum(self.infer_ms) / len(self.infer_ms) if self.infer_ms else 0.0
        return {"frames": self.frames, "unique": len(self.id_class), "down": self.down, "up": self.up,
                "per_class": self.per_class(), "max_concurrency": self.max_conc,
                "fps": round(1000.0 / ms, 1) if ms else 0.0,
                "people_summary": summarize(list(self.attrs.values())) if self.attrs else {}}

    # --- drawing: boxes with corner brackets, ID labels, motion trails, tripwire ---
    def draw(self, frame: np.ndarray, tracks: list[dict], hud: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        line_y = int(h * self.line_pct / 100)
        # tripwire
        col = SAGE if self.flash > 0 else GOLD
        for x in range(0, w, 22):
            cv2.line(frame, (x, line_y), (min(x + 12, w), line_y), col, 3 if self.flash > 0 else 2, cv2.LINE_AA)
        if self.flash > 0:
            self.flash -= 1
        for t in tracks:
            color = PALETTE[t["id"] % len(PALETTE)]
            pts = list(self.trails.get(t["id"], []))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(i / len(pts) * 3)), cv2.LINE_AA)
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            L = max(8, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
            for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(frame, (px, py), (px + dx * L, py), color, 3, cv2.LINE_AA)
                cv2.line(frame, (px, py), (px, py + dy * L), color, 3, cv2.LINE_AA)
            label = f"ID {t['id']} | {t['cls']} {t['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8
            cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 8, ty + 4), DEEP_TEAL, -1)
            cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 8, ty + 4), color, 1)
            cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GOLD, 1, cv2.LINE_AA)
            if t.get("attrs"):
                atxt = t["attrs"].replace("·", "|")
                (aw, ah), _ = cv2.getTextSize(atxt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                ay = min(frame.shape[0] - 4, y2 + ah + 8)
                cv2.rectangle(frame, (x1, ay - ah - 5), (x1 + aw + 8, ay + 4), DEEP_TEAL, -1)
                cv2.putText(frame, atxt, (x1 + 4, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.4, SAGE, 1, cv2.LINE_AA)
        if hud:
            txt = f"DOWN {self.down}   UP {self.up}   TRACKS {len(tracks)}   [{self.det.kind.upper()}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (10, 10), (24 + tw, 20 + th + 8), DEEP_TEAL, -1)
            cv2.rectangle(frame, (10, 10), (24 + tw, 20 + th + 8), GOLD, 1)
            cv2.putText(frame, txt, (17, 18 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SAGE, 1, cv2.LINE_AA)
        return frame


# ───────────────────────── helpers ─────────────────────────
def fit_width(frame: np.ndarray, max_w: int = MAX_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / w
        w, h = max_w, int(h * s)
    w, h = w - w % 2, h - h % 2  # H.264 needs even dimensions
    return cv2.resize(frame, (w, h)) if (w, h) != frame.shape[1::-1] else frame


def make_heatmap(heat: np.ndarray, base: np.ndarray, line_pct: int, title: str) -> np.ndarray:
    """Blend an accumulated presence map over a darkened still, using the site palette."""
    h, w = base.shape[:2]
    hm = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, w / 160), sigmaY=max(3, w / 160))
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    n = hm / hm.max() if hm.max() > 0 else hm
    n = np.sqrt(np.clip(n, 0, 1))
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    lut = np.array([(106, 74, 20), (134, 122, 60), (138, 205, 227), (235, 250, 255)], dtype=float)  # BGR: teal -> sage -> gold -> pale
    color = np.stack([np.interp(n, stops, lut[:, c]) for c in range(3)], axis=-1)
    dark = (base.astype(float) * 0.35 + np.array(DEEP_TEAL) * 0.65)
    a = np.clip(n * 1.1, 0, 0.92)[..., None]
    img = (dark * (1 - a) + color * a).astype(np.uint8)
    ly = int(h * line_pct / 100)
    for x in range(0, w, 24):
        cv2.line(img, (x, ly), (min(x + 12, w), ly), GOLD, 2, cv2.LINE_AA)
    cv2.putText(img, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SAGE, 1, cv2.LINE_AA)
    return img


def encode_h264(src: Path, dst: Path) -> bool:
    """Re-encode OpenCV's mp4v output to browser-playable H.264. Falls back to a plain copy."""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe:
        try:
            subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
                           check=True, timeout=600)
            return dst.exists() and dst.stat().st_size > 0
        except Exception:
            pass
    shutil.copyfile(src, dst)
    return False


def heading(dx: float, dy: float, new: bool = False) -> str:
    """Movement direction of a track between two frames (image coordinates: +y is down)."""
    if new:
        return "new"
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return "stationary"
    v = ("down" if dy > 0 else "up") if abs(dy) >= 0.5 else ""
    h = ("right" if dx > 0 else "left") if abs(dx) >= 0.5 else ""
    return "-".join(x for x in (v, h) if x)


CSV_HEADER = ["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "centroid_x", "centroid_y",
              "crossed_line", "direction", "heading", "attributes"]


# ───────────────────────── video job ─────────────────────────
MAX_FRAMES_HARD = int(os.environ.get("RCV_MAX_FRAMES", "0"))   # 0 = no limit: the whole video is always processed


def _vehicle_crops(store: dict, tid: int, score: float, frame: np.ndarray, box, keep: int = 4) -> None:
    """Remember the best few crops (big and confident) of a vehicle track for the type / model step."""
    lst = store.setdefault(tid, [])
    if len(lst) >= keep and score <= lst[-1][0]:
        return
    c = vehicles.best_crop(frame, box)
    if c is None:
        return
    lst.append((score, c))
    lst.sort(key=lambda x: -x[0])
    del lst[keep:]


def identify_vehicles(crops_by_root: dict, base: dict, extra: list[str], notes: list[str]) -> dict:
    """Type / model / class correction for each vehicle track. Returns {} (with a note) while CLIP is still preparing."""
    if not crops_by_root:
        return {}
    clf = vehicles.get_classifier()
    if clf is None:
        notes.append("Vehicle type / model recognition is still preparing in the background (one-time download of the CLIP model, "
                     "first use only), so this run shows the standard class names only. Run it again in a few minutes.")
        return {}
    try:
        return clf.classify({k: [c for _, c in v] for k, v in crops_by_root.items()}, base, extra)
    except Exception as e:  # noqa: BLE001
        notes.append("Vehicle recognition failed (" + (str(e)[:100] or e.__class__.__name__) + ") - standard class names are shown.")
        return {}


def run_video_job(job: dict, in_path: Path, out_dir: Path, on_progress: Callable[[float], None]) -> None:
    """
    Process an uploaded video exactly once, start to finish:

      pass 1  detect + track every frame of the whole video (no frame limit), then stop reading;
      clean   drop flicker, merge doubles, one stable class per object (see consolidate);
      vehicles  optional CLIP type / model for each car, truck, bus, motorcycle, bicycle ...;
      pass 2  draw the final boxes, count line crossings, write the video, heatmap and CSV.
    Writes tracked.mp4 / heatmap.png / tracks.csv into out_dir and fills `job`.
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise ValueError("Could not open this video. Please upload an MP4, MOV or WEBM file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(job.get("max_frames") or 0) or MAX_FRAMES_HARD or 0          # 0 = whole video
    n_target = min(limit, total) if (limit and total) else (limit or total or 0)

    ok, first = cap.read()
    if not ok:
        raise ValueError("The video has no readable frames.")
    first = fit_width(first)
    H, W = first.shape[:2]

    det = TrackingSession(job["model"], job["tracker"], job["conf"], job["classes"], job["line"], job.get("imgsz", 640),
                          job.get("vocab"), False)
    names = det.names
    notes: list[str] = []
    want_vehicles = bool(job.get("vehicles", True))
    raw: list[np.ndarray] = []
    crops: dict[int, list] = {}
    frame, idx = first, 0
    try:
        while True:                                   # ---- pass 1: the whole video, once ----
            idx += 1
            rows = det.det.track(frame)
            raw.append(np.array(rows, dtype=float).reshape(-1, 7))
            if want_vehicles:
                for x1, y1, x2, y2, tid, cf, cid in raw[-1]:
                    if names.get(int(cid), "") in vehicles.BASE_CLASSES:
                        _vehicle_crops(crops, int(tid), float(cf) * (x2 - x1) * (y2 - y1), frame, (x1, y1, x2, y2))
            on_progress(min(0.78, (idx / n_target if n_target else 1 - 1 / (1 + idx / 200)) * 0.78))
            if limit and idx >= limit:
                break
            ok, frame = cap.read()
            if not ok:                                # end of the video: detection stops here
                break
            frame = fit_width(frame)
            if frame.shape[1::-1] != (W, H):
                frame = cv2.resize(frame, (W, H))
    finally:
        cap.release()
    if not raw:
        raise ValueError("The video has no readable frames.")

    on_progress(0.80)
    finals, info = consolidate(raw, names, fps, job["conf"])          # ---- clean-up ----
    st_cls: dict[int, str] = {}
    for fr in finals:
        for t in fr:
            st_cls[t["id"]] = t["cls"]
    details: dict[int, str] = {}
    if want_vehicles and crops:                                       # ---- vehicle type / model ----
        by_new: dict[int, list] = {}
        raw_to_new = info["raw_to_new"]                           # tracker id -> final id (merged doubles share one)
        for rid, lst in crops.items():
            nid = raw_to_new.get(rid)
            if nid and st_cls.get(nid) in vehicles.BASE_CLASSES:
                by_new.setdefault(nid, []).extend(lst)
        for nid in by_new:
            by_new[nid] = sorted(by_new[nid], key=lambda x: -x[0])[:4]
        res = identify_vehicles(by_new, {k: st_cls[k] for k in by_new}, list(job.get("extra") or []), notes)
        for nid, r in res.items():
            if r.get("label"):
                details[nid] = r["label"]
            if r.get("class_override"):
                for fr in finals:
                    for t in fr:
                        if t["id"] == nid:
                            t["cls"] = r["class_override"]
                st_cls[nid] = r["class_override"]
    on_progress(0.85)

    # ---- pass 2: draw / count / export the cleaned result ----
    cid_of: dict[str, int] = {}
    for c in st_cls.values():
        cid_of.setdefault(c, len(cid_of))
    sess = TrackingSession(job["model"], job["tracker"], job["conf"], job["classes"], job["line"], job.get("imgsz", 640),
                           None, bool(job.get("people")), render_only=True)
    sess.names = {v: k for k, v in cid_of.items()}
    sess.details = details
    raw_path, final_path = out_dir / "raw.mp4", out_dir / "tracked.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    heat = np.zeros((H // 4 + 1, W // 4 + 1), dtype=np.float32)
    ledger: list[dict] = []
    prev_pos: dict[int, tuple] = {}
    timeline: list[tuple] = []
    csv_f = open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow(CSV_HEADER)
    cap = cv2.VideoCapture(str(in_path))
    try:
        for i, fr in enumerate(finals, start=1):
            if i == 1:
                frame = first.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = fit_width(frame)
                if frame.shape[1::-1] != (W, H):
                    frame = cv2.resize(frame, (W, H))
            rows = np.array([[*t["box"], t["id"], t["conf"], cid_of[t["cls"]]] for t in fr], dtype=float).reshape(-1, 7)
            tracks = sess.process(rows, frame)
            for t in tracks:
                x1, y1, x2, y2 = t["box"]
                cx, cy = int(t["cx"]), int(t["cy"])
                heat[min(max(cy // 4, 0), heat.shape[0] - 1), min(max(cx // 4, 0), heat.shape[1] - 1)] += 1
                px, py = prev_pos.get(t["id"], (cx, cy))
                cw.writerow([i, t["id"], t["cls"], round(t["conf"], 2), int(x1), int(y1), int(x2), int(y2), cx, cy,
                             "true" if t["crossed"] else "false", t["direction"] or "none",
                             heading(cx - px, cy - py, t["id"] not in prev_pos), t.get("attrs", "")])
                if t["id"] not in prev_pos and len(ledger) < 300:   # one row per tracked object (first sighting), so every bike / truck / person is listed
                    ledger.append({"frame": i, "id": t["id"], "class": t["cls"], "conf": round(t["conf"], 2),
                                   "cx": cx, "cy": cy, "crossed": t["crossed"], "direction": t["direction"],
                                   "dx": round(cx - px, 1), "dy": round(cy - py, 1),
                                   "bbox": [int(x1), int(y1), int(x2), int(y2)], "attrs": t.get("attrs", "")})
                prev_pos[t["id"]] = (cx, cy)
                timeline.append((i, t["id"], t["cls"], cx, cy, bool(t["crossed"]), t["direction"], t.get("attrs", "")))
            writer.write(sess.draw(frame.copy(), tracks))
            on_progress(0.85 + 0.10 * i / len(finals))
    finally:
        cap.release()
        writer.release()
        csv_f.close()

    on_progress(0.96)
    encode_h264(raw_path, final_path)
    raw_path.unlink(missing_ok=True)
    st = sess.stats()
    st.update({"frames": len(raw), "raw_tracks": info["raw_tracks"], "removed_short": info["removed_short"],
               "merged_duplicates": info["merged"], "vehicle_details": len(details), "notes": notes,
               "whole_video": not limit or len(raw) < (limit or 0)})
    title = f"ROYAL COURT HEATMAP | {job['tracker'].upper()} | {job['model']} | UNIQUE {st['unique']}"
    cv2.imwrite(str(out_dir / "heatmap.png"), make_heatmap(heat, first, job["line"], title))
    rep = traffic_report.build_report(timeline, fps, W, H, job["line"], int(job.get("report_bin") or 0), str(job.get("expected_dir") or ""),
                                      float(job.get("stop_seconds") or 10), len(raw), notes)
    (out_dir / "report.html").write_text(traffic_report.to_html(rep, out_dir / "heatmap.png"), encoding="utf-8")
    (out_dir / "report.csv").write_text(traffic_report.to_csv(rep), encoding="utf-8")
    st["report"] = rep
    job["stats"] = st
    job["ledger"] = ledger
    job["width"], job["height"] = W, H


# ───────────────────────── image mode ─────────────────────────
def _nms(dets: np.ndarray, thr: float = 0.5, agnostic: bool = False) -> np.ndarray:
    """Non-maximum suppression (numpy): per class, or across classes when `agnostic` (one name per object)."""
    keep = []
    for c in ([None] if agnostic else np.unique(dets[:, 5])):
        d = dets if agnostic else dets[dets[:, 5] == c]
        d = d[np.argsort(-d[:, 4])]
        while len(d):
            keep.append(d[0])
            d = d[1:][iou_batch(d[:1, :4], d[1:, :4])[0] < thr] if len(d) > 1 else d[1:]
    return np.array(keep, dtype=float).reshape(-1, 6)


def _boxes(r) -> np.ndarray:
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6))
    return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]])


def detect_tiled(model, img, conf, ids, imgsz, tile=640, overlap=0.25, max_tiles=24, agnostic=False) -> np.ndarray:
    """Tiny-object mode (SAHI idea): one pass on the whole image + overlapping tiles, merged with NMS."""
    H, W = img.shape[:2]
    while True:
        step = int(tile * (1 - overlap))
        xs = list(range(0, max(1, W - tile + step), step)) or [0]
        ys = list(range(0, max(1, H - tile + step), step)) or [0]
        if len(xs) * len(ys) <= max_tiles:
            break
        tile = int(tile * 1.25)
    with INFER_LOCK:
        all_d = [_boxes(model.predict(img, conf=conf, classes=ids, imgsz=imgsz, agnostic_nms=agnostic, verbose=False)[0])]
        tiles, offs = [], []
        for y in ys:
            for x in xs:
                x2, y2 = min(W, x + tile), min(H, y + tile)
                tiles.append(img[max(0, y2 - tile):y2, max(0, x2 - tile):x2])
                offs.append((max(0, x2 - tile), max(0, y2 - tile)))
        for res, (ox, oy) in zip(model.predict(tiles, conf=conf, classes=ids, imgsz=max(imgsz, tile), agnostic_nms=agnostic, verbose=False), offs):
            d = _boxes(res)
            if len(d):
                d[:, [0, 2]] += ox
                d[:, [1, 3]] += oy
                all_d.append(d)
    d = np.vstack(all_d)
    return _nms(d, 0.5, agnostic) if len(d) else d


def run_image_job(job: dict, in_path: Path, out_dir: Path) -> None:
    """Detection on a single image (no tracking): annotated JPEG, density heatmap, CSV, optional people attributes."""
    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError("Could not read this image. Please upload a JPG, PNG or WEBP file.")
    tiles = bool(job.get("tiles"))
    cap = 2560 if tiles else 1600
    if max(img.shape[:2]) > cap:
        s = cap / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    H, W = img.shape[:2]
    auto = bool(job.get("auto"))
    extra = list(job.get("extra") or [])          # names typed in the "find anything by name" box
    chosen = [c.lower() for c in job.get("classes", [])] if not auto else []
    if not auto and not chosen and not extra:
        raise ValueError("Pick at least one target in the menus, or switch to Auto to detect everything.")
    wanted = None if auto else set(chosen) | set(extra)   # applied AFTER detection, so nothing is forced into a wrong class
    notes: list[str] = []

    t0 = time.perf_counter()
    agnostic = False
    conf = job["conf"]
    # open-vocabulary results are checked crop by crop with CLIP (see verify.py); the detector only proposes boxes
    verifier = None
    verifier_wanted = False
    if not FAKE and (auto or extra or any(c not in COCO_SET for c in chosen)):
        verifier = verify.get_verifier(MODELS_DIR)
        verifier_wanted = verifier is not None
        if verifier is None:
            conf = max(conf, 0.45)
            notes.append("The CLIP checking model is still being prepared in the background (one-time), so this run used a strict "
                         "confidence of 0.45 without the second check. Run again in a few minutes for much more accurate names.")
    if FAKE:
        names = Detector.FAKE_NAMES
        dets = np.array([[W * .1, H * .3, W * .2, H * .8, .91, 0], [W * .45, H * .5, W * .75, H * .85, .87, 2],
                         [W * .8, H * .35, W * .88, H * .85, .66, 0]], dtype=float)
    else:
        outside_coco = [c for c in chosen if c not in COCO_SET]
        scenery = [c for c in outside_coco if c not in DETECT_SET]      # e.g. "forest": not in the Auto list, needs its own search
        if extra or scenery:                       # typed / scenery names: build a vocabulary for this request
            vocab = list(dict.fromkeys((DETECT_VOCAB if auto else chosen) + extra))
            model = get_shared_model(WORLD_DEFAULT, vocab)
            agnostic, conf = True, 0.15 if verifier_wanted else max(0.25, conf * 0.85)
        elif (auto or outside_coco) and universal_ready():
            model = get_universal_model()          # knows ~2,100 everyday things (lion, pigeon, whiteboard ...)
            agnostic, conf = True, 0.15 if verifier_wanted else max(0.25, conf * 0.85)
        else:
            model = get_shared_model(job["model"])  # standard 80-class detector (always instant)
            if auto or outside_coco:
                start_warmup()
                notes.append("The wide-vocabulary model is still being prepared in the background (one-time, first run only), "
                             "so the standard 80-class detector was used. Run again in a few minutes for the full vocabulary.")
        names = model.names
        imgsz = job.get("imgsz", 960)
        if tiles:
            dets = detect_tiled(model, img, conf, None, imgsz, agnostic=agnostic)
        else:
            with INFER_LOCK:
                dets = _boxes(model.predict(img, conf=conf, classes=None, imgsz=imgsz, agnostic_nms=agnostic, verbose=False)[0])
    if verifier is not None and len(dets):
        dets = dets[(dets[:, 2] - dets[:, 0] >= 10) & (dets[:, 3] - dets[:, 1] >= 10)]
        cand = list(dict.fromkeys(DETECT_VOCAB + [c for c in chosen if c not in DETECT_SET] + extra))
        crops, keep_i = [], []
        for i, d in enumerate(dets):
            c = verify.crop_with_context(img, d[:4])
            if c is not None:
                crops.append(c)
                keep_i.append(i)
        res = verifier.verify(crops, cand, [names.get(int(dets[i, 5]), "") for i in keep_i])
        rows = []
        grad = verify.gradient_map(img)
        ref = verify.edge_ref(grad)
        floor = max(float(job["conf"]), verify.AUTO_FLOOR) if (auto or extra) else float(job["conf"])
        dropped_blur = 0
        for i, r in zip(keep_i, res):
            if r is None:
                continue
            label, p = r
            if label not in vehicles.BASE_CLASSES and label != "person" and verify.crispness(grad, ref, dets[i, :4]) < verify.BLUR_BELOW:
                dropped_blur += 1                      # out-of-focus blob / light circle, not an object
                continue
            if float(np.sqrt(max(dets[i, 4], 1e-6) * p)) < floor:
                continue                               # too unsure to show in Auto mode
            rows.append((i, label, p))
        if dropped_blur:
            notes.append(f"{dropped_blur} out-of-focus blob(s) (light circles, blurred background) were ignored.")
        lab_id = {l: k for k, l in enumerate(dict.fromkeys(r[1] for r in rows))}
        names = {v: k for k, v in lab_id.items()}
        dets = np.array([[*dets[i, :4], float(np.sqrt(max(dets[i, 4], 1e-6) * p)), lab_id[l]] for i, l, p in rows], dtype=float).reshape(-1, 6)
    if wanted is not None and len(dets):
        dets = dets[[names.get(int(c), str(int(c))).lower() in wanted for c in dets[:, 5]]]
    dets = filter_dets(dets, names, W, H, tiles)          # no specks, one box per object
    ms = (time.perf_counter() - t0) * 1000
    dets = dets[np.argsort(-dets[:, 4])] if len(dets) else dets

    # optional: vehicle type / model (CLIP) - also corrects a car that was called a truck, and so on
    veh_tag: dict[int, str] = {}
    cls_fix: dict[int, str] = {}
    if job.get("vehicles", True) and len(dets):
        vc: dict[int, list] = {}
        for i, d in enumerate(dets):
            if names.get(int(d[5]), "") in vehicles.BASE_CLASSES:
                c = vehicles.best_crop(img, d[:4])
                if c is not None:
                    vc[i] = [(1.0, c)]
        base = {i: names.get(int(dets[i, 5]), "") for i in vc}
        for i, r in identify_vehicles(vc, base, list(job.get("extra") or []), notes).items():
            if r.get("label"):
                veh_tag[i] = r["label"]
            if r.get("class_override"):
                cls_fix[i] = r["class_override"]

    # optional: people attributes (gender / age group / bag / phone / activity)
    attrs: dict[int, dict] = {}
    if job.get("people") and not get_analyzer().is_ready():
        get_analyzer().load_async()
        notes.append("People analysis is downloading its model in the background (one-time, first use only) - "
                     "this run shows detections without person details. Run it again in a few minutes.")
    elif job.get("people") and len(dets):
        idx, crops = [], []
        for i, d in enumerate(dets):
            if names.get(int(d[5]), "") == "person":
                c = person_crop(img, d)
                if c is not None:
                    idx.append(i)
                    crops.append(c)
        for i, a in zip(idx, get_analyzer().analyze(crops)):
            attrs[i] = a

    ann, heat = img.copy(), np.zeros((H // 4 + 1, W // 4 + 1), dtype=np.float32)
    th_line = max(2, W // 700)
    fs = max(0.5, min(1.1, W / 1600))
    per_class: dict = {}
    rows, ledger = [], []
    for i, (x1, y1, x2, y2, conf, cid) in enumerate(dets, start=1):
        cls = cls_fix.get(i - 1) or names.get(int(cid), str(int(cid)))
        per_class[cls] = per_class.get(cls, 0) + 1
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tag = attrs.get(i - 1, {}).get("tag", "") or veh_tag.get(i - 1, "")
        tag_ui = tag
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, th_line, cv2.LINE_AA)
        L = max(6, min(24, (x2 - x1) // 4, (y2 - y1) // 4))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(ann, (px, py), (px + dx * L, py), color, th_line + 2, cv2.LINE_AA)
            cv2.line(ann, (px, py), (px, py + dy * L), color, th_line + 2, cv2.LINE_AA)
        label = f"#{i} {cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = y1 - 6 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(ann, (x1, ty - th - 5), (x1 + tw + 8, ty + 4), DEEP_TEAL, -1)
        cv2.rectangle(ann, (x1, ty - th - 5), (x1 + tw + 8, ty + 4), color, 1)
        cv2.putText(ann, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, GOLD, 1, cv2.LINE_AA)
        if tag:
            tag = tag.replace("·", "|")  # OpenCV's font is ASCII only
            (aw, ah), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, 1)
            ay = min(H - 4, y2 + ah + 8)
            cv2.rectangle(ann, (x1, ay - ah - 5), (x1 + aw + 8, ay + 4), DEEP_TEAL, -1)
            cv2.putText(ann, tag, (x1 + 4, ay), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, SAGE, 1, cv2.LINE_AA)
        area_w = max(1.0, (x2 - x1) * (y2 - y1) / (W * H) * 40)
        heat[min(cy // 4, heat.shape[0] - 1), min(cx // 4, heat.shape[1] - 1)] += area_w + 1
        rows.append([i, i, cls, round(float(conf), 2), x1, y1, x2, y2, cx, cy, "false", "none", "n/a", tag_ui])
        ledger.append({"frame": 1, "id": i, "class": cls, "conf": round(float(conf), 2), "cx": cx, "cy": cy,
                       "crossed": False, "direction": None, "dx": 0, "dy": 0, "bbox": [x1, y1, x2, y2], "attrs": tag_ui})
    cv2.imwrite(str(out_dir / "annotated.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / "heatmap.png"), make_heatmap(heat, img, 50, f"DETECTION DENSITY | {job['model']} | OBJECTS {len(dets)}"))
    with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(CSV_HEADER)
        cw.writerows(rows)
    avg = float(np.mean(dets[:, 4])) if len(dets) else 0.0
    job["stats"] = {"frames": 1, "unique": len(dets), "down": 0, "up": 0, "per_class": per_class,
                    "max_concurrency": len(dets), "fps": round(1000 / ms, 1) if ms else 0.0,
                    "avg_conf": round(avg * 100, 1), "infer_ms": round(ms, 1), "classes_found": len(per_class),
                    "people_summary": summarize(list(attrs.values())) if attrs else {}, "tiled": tiles,
                    "mode": "auto" if auto else "selected", "notes": notes}
    job["ledger"] = ledger[:60]
    job["width"], job["height"] = W, H
