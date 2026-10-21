"""
SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Own implementation (numpy + scipy only, no filterpy):
  * one Kalman filter per track, state = [cx, cy, s(area), r(aspect), vx, vy, vs]
  * IoU cost matrix between the *predicted* track boxes and new detections
  * Hungarian algorithm (scipy.optimize.linear_sum_assignment) for the matching

Detections are rows of [x1, y1, x2, y2, conf, cls]. update() returns rows of
[x1, y1, x2, y2, track_id, conf, cls].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4), both [x1,y1,x2,y2]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def _to_z(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def _to_box(x):
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for one bounding box."""

    def __init__(self, det, track_id: int):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1))
        self.x[:4] = _to_z(det[:4])
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.conf = float(det[4])
        self.cls = int(det[5])

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:  # area would go negative
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _to_box(self.x)

    def update(self, det):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = float(det[4])
        self.cls = int(det[5])
        z = _to_z(det[:4])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self):
        return _to_box(self.x)


class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1  # ids are per-instance, so each job/session starts at 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        dets = np.zeros((0, 6)) if dets is None or len(dets) == 0 else np.asarray(dets, dtype=float)

        # 1. predict every existing track one step forward
        preds = []
        for t in list(self.trackers):
            p = t.predict()
            if np.any(np.isnan(p)):
                self.trackers.remove(t)
            else:
                preds.append(p)
        pred_boxes = np.array(preds).reshape(-1, 4)

        # 2. associate: IoU matrix + Hungarian algorithm
        matched, unmatched_dets = [], list(range(len(dets)))
        if len(pred_boxes) and len(dets):
            iou = iou_batch(pred_boxes, dets[:, :4])
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(dets)) if d not in cols]
            for r, c in zip(rows, cols):
                if iou[r, c] < self.iou_threshold:
                    unmatched_dets.append(c)
                else:
                    matched.append((r, c))

        # 3. update matched tracks, spawn new tracks for unmatched detections
        for r, c in matched:
            self.trackers[r].update(dets[c])
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d], self._next_id))
            self._next_id += 1

        # 4. emit confirmed tracks, drop the dead ones
        out = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                b = t.box
                out.append([b[0], b[1], b[2], b[3], t.id, t.conf, t.cls])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(out, dtype=float).reshape(-1, 7)
