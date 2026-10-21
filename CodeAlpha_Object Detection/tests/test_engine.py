"""Offline tests: run with  RCV_FAKE_DETECTOR=1 pytest -q   (no internet / GPU / YOLO weights needed)."""
import os, sys
os.environ["RCV_FAKE_DETECTOR"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from pathlib import Path

import cv2
import numpy as np
import pytest

import engine
from sort_tracker import Sort, iou_batch


def test_iou():
    a = np.array([[0, 0, 10, 10.]]); b = np.array([[5, 5, 15, 15.], [20, 20, 30, 30.]])
    r = iou_batch(a, b)
    assert abs(r[0, 0] - 25 / 175) < 1e-6 and r[0, 1] == 0


def test_sort_keeps_identity():
    s = Sort(min_hits=1)
    ids = set()
    for k in range(20):
        out = s.update(np.array([[10 + 4 * k, 10, 60 + 4 * k, 110, .9, 0.]]))
        ids |= set(out[:, 4].astype(int))
    assert ids == {1}


def test_sort_new_object_gets_new_id():
    s = Sort(min_hits=1)
    s.update(np.array([[0, 0, 50, 50, .9, 0.]]))
    out = s.update(np.array([[0, 0, 50, 50, .9, 0.], [200, 200, 260, 260, .8, 2.]]))
    assert sorted(out[:, 4].astype(int)) == [1, 2]


def _video(path, n=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for _ in range(n):
        w.write(np.full((360, 640, 3), 60, np.uint8))
    w.release()


def test_video_job(tmp_path):
    _video(tmp_path / "in.mp4")
    job = dict(max_frames=0, model="yolov8n.pt", tracker="sort", conf=.35, classes=["person", "car"], line=50)
    engine.run_video_job(job, tmp_path / "in.mp4", tmp_path, lambda p: None)
    assert job["stats"]["frames"] == 40 and job["stats"]["unique"] >= 1   # 0 = the whole video, then it stops
    for f in ("tracked.mp4", "heatmap.png", "tracks.csv"):
        assert (tmp_path / f).stat().st_size > 0
    assert len(job["ledger"]) <= 300


def test_image_job(tmp_path):
    cv2.imwrite(str(tmp_path / "in.jpg"), np.full((300, 500, 3), 90, np.uint8))
    job = dict(model="yolov8n.pt", conf=.35, classes=["person", "car"])
    engine.run_image_job(job, tmp_path / "in.jpg", tmp_path)
    assert job["stats"]["unique"] == 3 and (tmp_path / "annotated.jpg").exists()


def test_bad_video(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    job = dict(max_frames=10, model="yolov8n.pt", tracker="sort", conf=.35, classes=["person"], line=50)
    with pytest.raises(ValueError):
        engine.run_video_job(job, tmp_path / "bad.mp4", tmp_path, lambda p: None)


def test_image_people_and_nms(tmp_path):
    cv2.imwrite(str(tmp_path / "in.jpg"), np.random.randint(0, 255, (400, 600, 3), np.uint8))
    job = dict(model="yolov8n.pt", conf=.35, classes=["person", "car"], imgsz=640, vocab=None, tiles=False, people=True)
    engine.run_image_job(job, tmp_path / "in.jpg", tmp_path)
    assert job["stats"]["people_summary"] and job["ledger"][0]["attrs"]
    d = np.array([[0, 0, 100, 100, .9, 0], [5, 5, 102, 98, .8, 0], [300, 300, 350, 350, .7, 1]], float)
    assert len(engine._nms(d)) == 2


def test_video_job_frame_limit_still_works(tmp_path):
    _video(tmp_path / "in.mp4")
    job = dict(max_frames=12, model="yolov8n.pt", tracker="sort", conf=.35, classes=["person", "car"], line=50)
    engine.run_video_job(job, tmp_path / "in.mp4", tmp_path, lambda p: None)
    assert job["stats"]["frames"] == 12
