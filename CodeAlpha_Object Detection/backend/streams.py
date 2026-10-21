"""RTSP / IP-camera ingestion: a background thread reads frames with OpenCV, tracks them,
and publishes the latest annotated JPEG (served to the browser as an MJPEG stream)."""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Optional

import cv2

from engine import TrackingSession, fit_width

MAX_SESSIONS = int(os.environ.get("RCV_MAX_STREAMS", "2"))
MAX_SECONDS = int(os.environ.get("RCV_STREAM_MAX_SECONDS", "0"))  # 0 = no time limit
IDLE_SECONDS = int(os.environ.get("RCV_STREAM_IDLE_SECONDS", "120"))  # only stops once the browser has gone away


class StreamSession:
    def __init__(self, url: str, transport: str, params: dict):
        self.id = "st_" + uuid.uuid4().hex[:8]
        self.url = url
        self.transport = transport
        self.params = params
        self.latest: Optional[bytes] = None
        self.stats: dict = {"frames": 0, "unique": 0, "down": 0, "up": 0, "per_class": {}, "fps": 0.0, "tracks": 0}
        self.error: Optional[str] = None
        self.opened = threading.Event()
        self.stop_flag = threading.Event()
        self.last_seen = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout: float = 15.0) -> None:
        self.thread.start()
        self.opened.wait(timeout)
        if self.error:
            raise ConnectionError(self.error)
        if not self.opened.is_set():
            self.stop_flag.set()
            raise ConnectionError("Timed out while opening the stream. Check the URL, network and camera credentials.")

    def stop(self) -> None:
        self.stop_flag.set()

    def _open(self):
        # tcp is the reliable default; udp has lower latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{'udp' if self.transport == 'udp' else 'tcp'}|stimeout;8000000"
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _run(self) -> None:
        cap = None
        try:
            cap = self._open()
            if not cap.isOpened():
                self.error = "Could not open the stream. Check the URL, credentials and that the camera is reachable from this server."
                self.opened.set()
                return
            sess = TrackingSession(**self.params)
            self.opened.set()
            started = time.time()
            fails = 0
            while not self.stop_flag.is_set():
                if (MAX_SECONDS and time.time() - started > MAX_SECONDS) or time.time() - self.last_seen > IDLE_SECONDS:
                    break
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        self.error = "The stream stopped sending frames."
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                frame = fit_width(frame, 960)
                tracks = sess.step(frame)
                sess.draw(frame, tracks)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    self.latest = buf.tobytes()
                st = sess.stats()
                st["tracks"] = len(tracks)
                fh, fw = frame.shape[:2]
                st["live_tracks"] = [{"id": t["id"], "cls": t["cls"], "conf": round(t["conf"], 2),
                                      "box": [t["box"][0] / fw, t["box"][1] / fh, t["box"][2] / fw, t["box"][3] / fh],
                                      "direction": t["direction"], "attrs": t.get("attrs", "")} for t in tracks]
                self.stats = st
        except Exception as e:  # noqa: BLE001
            self.error = f"Stream error: {e}"
            self.opened.set()
        finally:
            if cap is not None:
                cap.release()
            self.stop_flag.set()


SESSIONS: dict[str, StreamSession] = {}
_LOCK = threading.Lock()


def create_session(url: str, transport: str, params: dict) -> StreamSession:
    with _LOCK:
        for sid in [k for k, v in SESSIONS.items() if v.stop_flag.is_set()]:
            SESSIONS.pop(sid, None)
        if len(SESSIONS) >= MAX_SESSIONS:
            raise RuntimeError("Too many live streams are running on this server. Disconnect one and try again.")
        s = StreamSession(url, transport, params)
        SESSIONS[s.id] = s
    try:
        s.start()
    except Exception:
        with _LOCK:
            SESSIONS.pop(s.id, None)
        raise
    return s
