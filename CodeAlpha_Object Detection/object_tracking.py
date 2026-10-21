#!/usr/bin/env python3
"""
Task 4 - Object Detection and Tracking (command-line, real-time OpenCV window).

    python object_tracking.py                              # webcam
    python object_tracking.py --source video.mp4           # video file
    python object_tracking.py --source rtsp://user:pw@ip/stream
    python object_tracking.py --source video.mp4 --save out.mp4 --tracker sort
    python object_tracking.py --classes person,car --conf 0.4 --line 60

Pipeline: OpenCV capture -> YOLOv8 detection -> SORT (Kalman filter + Hungarian matching),
ByteTrack or BoT-SORT -> boxes, labels and persistent IDs drawn in real time. Press q / ESC to quit.
The same engine powers the website in backend/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import cv2  # noqa: E402

from engine import TrackingSession, fit_width  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 + SORT/ByteTrack/BoT-SORT real-time tracking")
    ap.add_argument("--source", default="0", help="webcam index (0), video file or rtsp:// URL")
    ap.add_argument("--model", default="yolov8n.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"])
    ap.add_argument("--tracker", default="sort", choices=["sort", "bytetrack", "botsort"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", default="person,car", help="comma separated COCO class names")
    ap.add_argument("--line", type=int, default=50, help="counting line position, % of frame height")
    ap.add_argument("--save", default="", help="write the annotated video to this path")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (headless)")
    ap.add_argument("--imgsz", type=int, default=640, choices=[480, 640, 960, 1280], help="detection resolution")
    ap.add_argument("--max-width", type=int, default=960)
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {a.source}")
    sess = TrackingSession(a.model, a.tracker, a.conf, [c.strip() for c in a.classes.split(",")], a.line, a.imgsz)
    writer, t_prev, fps_s = None, time.time(), 0.0
    print("Tracking... press q or ESC in the window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_width(frame, a.max_width)
        tracks = sess.step(frame)
        sess.draw(frame, tracks)
        now = time.time()
        fps_s = 0.9 * fps_s + 0.1 / max(now - t_prev, 1e-6) if fps_s else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"FPS {fps_s:4.1f}", (frame.shape[1] - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (138, 205, 227), 2, cv2.LINE_AA)
        if a.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(frame)
        if not a.no_display:
            cv2.imshow("The Royal Court of Vision - tracking", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cap.release()
    if writer:
        writer.release()
        print("Saved:", a.save)
    cv2.destroyAllWindows()
    st = sess.stats()
    print(f"Frames {st['frames']} | unique IDs {st['unique']} | crossed down {st['down']} / up {st['up']} | per class {st['per_class']}")


if __name__ == "__main__":
    main()
