"""
Download everything the website can ask for, ONCE, before you test - so nothing downloads while you wait.

    cd backend
    python prepare_models.py --all        # every detector in the model menu + wide vocabulary + CLIP (recommended, ~2 GB)
    python prepare_models.py              # the essentials only (yolov8n/s, wide vocabulary, CLIP)
    python prepare_models.py --check      # show what is already downloaded (downloads nothing)
    python prepare_models.py --smoke      # after downloading: run one bundled photo and time it
    python prepare_models.py --no-people  # skip the CLIP model (vehicle types / models + 'Analyze people', ~600 MB)

Everything is saved under backend/models, backend/weights and the Hugging Face cache, so the website never makes a
visitor wait. The server also does the essentials by itself in the background on start-up (RCV_NO_WARMUP=1 turns that off).
"""
import sys
import threading
import time
from pathlib import Path

import engine


def _hf_cached() -> bool:
    try:
        from transformers import CLIPModel
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    have = {n: (engine.MODELS_DIR / n).exists() for n in sorted(engine.ALLOWED_MODELS)}
    have[engine.UNIVERSAL_FILE + " (wide vocabulary)"] = engine.universal_ready()
    have["CLIP for YOLO-World text (backend/weights/clip)"] = engine.clip_ready()
    have["CLIP for vehicle models + people (Hugging Face cache)"] = _hf_cached()
    return have


def print_status() -> bool:
    st = status()
    for k, v in st.items():
        print(f"  [{'x' if v else ' '}] {k}")
    return all(st.values())


def fetch_all_detectors() -> None:
    for n in sorted(engine.ALLOWED_MODELS):
        if not (engine.MODELS_DIR / n).exists():
            engine.WARMUP["step"] = f"Detector {n}"
            (engine._load_world if "world" in n else engine._load_yolo)(n)


def smoke() -> None:
    import cv2
    import numpy as np  # noqa: F401
    import tempfile
    img = Path(__file__).resolve().parent.parent / "frontend" / "assets" / "photos" / "traffic.jpg"
    out = Path(tempfile.mkdtemp())
    for label, job in (("standard detector", dict(model="yolov8n.pt", conf=.35, classes=["person", "car"], imgsz=640, auto=False)),
                       ("auto / wide vocabulary", dict(model="yolov8n.pt", conf=.35, classes=[], imgsz=960, auto=True))):
        t = time.perf_counter()
        engine.run_image_job(job, img, out)
        print(f"  {label}: {job['stats']['unique']} objects in {time.perf_counter() - t:.1f}s -> {job['stats']['per_class']}")
        for n in job["stats"].get("notes", []):
            print("   note:", n)
    print("  annotated image:", out / "annotated.jpg")


def main() -> int:
    if "--check" in sys.argv:
        print("Downloaded so far:")
        return 0 if print_status() else 1
    people = "--no-people" not in sys.argv
    done = threading.Event()

    def show():
        while not done.is_set():
            st = engine.warmup_status()
            extra = f"  CLIP {int(st['clip_progress'] * 100)}%" if st["step"].startswith("Wide") else ""
            print(f"\r[{st['state']}] {st['step']}{extra}      ", end="", flush=True)
            time.sleep(2)

    threading.Thread(target=show, daemon=True).start()
    engine.WARMUP["state"] = "running"
    try:
        if "--all" in sys.argv:
            fetch_all_detectors()
        engine.warmup(people=people)
    finally:
        done.set()
    st = engine.warmup_status()
    print(f"\nfinished: {st['state']}" + (f" - {st['error']}" if st["error"] else ""))
    print("Downloaded now:")
    print_status()
    if "--smoke" in sys.argv and st["state"] == "ready":
        print("Smoke test:")
        smoke()
    return 0 if st["state"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
