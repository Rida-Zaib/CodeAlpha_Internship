"""API tests (need `pip install fastapi httpx python-multipart`). Run: RCV_FAKE_DETECTOR=1 pytest -q"""
import os, sys, time
os.environ["RCV_FAKE_DETECTOR"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pytest
pytest.importorskip("fastapi"); pytest.importorskip("httpx")
import cv2, numpy as np
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_video_flow(tmp_path):
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for _ in range(30):
        w.write(np.full((360, 640, 3), 60, np.uint8))
    w.release()
    r = client.post("/api/jobs", files={"file": ("v.mp4", p.read_bytes(), "video/mp4")},
                    data={"tracker": "sort", "model": "yolov8n.pt", "conf": "0.35", "classes": "person,car", "line": "50", "max_frames": "20"})
    jid = r.json()["id"]
    for _ in range(60):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    for ep in ("video", "heatmap", "csv", "rows"):
        assert client.get(f"/api/jobs/{jid}/{ep}").status_code == 200


def test_image_flow():
    ok, buf = cv2.imencode(".jpg", np.full((300, 400, 3), 90, np.uint8))
    r = client.post("/api/image", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")}, data={"classes": "person,car"})
    assert r.status_code == 200 and r.json()["stats"]["unique"] == 3


def test_rejects_bad_input():
    assert client.post("/api/jobs", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/stream/rtsp-connect", json={"url": "file:///etc/passwd"}).status_code == 400


def test_websocket_live():
    ok, buf = cv2.imencode(".jpg", np.full((360, 640, 3), 90, np.uint8))
    with client.websocket_connect("/ws/live?tracker=sort&classes=person,car") as ws:
        ws.send_bytes(buf.tobytes())
        d = ws.receive_json()
        assert "tracks" in d
