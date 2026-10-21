"""
Second opinion for open-vocabulary detections (Auto mode).

YOLO-World is good at *finding things* but weak at *naming* them among thousands of words (squirrel -> "beaver",
mushroom -> "tree trunk"). CLIP is much better at naming a single crop. So for every box:

  1. crop it (with a little context) and ask CLIP which of the ~2,000 names fits best,
     next to a handful of "not an object" answers (background, blurry foliage, a texture ...);
  2. the box gets CLIP's name; it is dropped when CLIP says "not an object", is unsure, or disagrees with the detector
     without being confident.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

NOT_AN_OBJECT = ["background", "blurry background", "bokeh", "out of focus foliage", "leaves", "plain texture", "ground", "a wall",
                 "empty space", "shadow", "part of an object", "sky", "water surface", "a pattern"]

ACCEPT = float(os.environ.get("RCV_VERIFY_ACCEPT", "0.50"))    # CLIP alone is this sure -> accept its name
AGREE = float(os.environ.get("RCV_VERIFY_AGREE", "0.15"))      # detector's name is in CLIP's top 3 and CLIP is at least this sure


def crop_with_context(img: np.ndarray, box, pad: float = 0.12, min_side: int = 28) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    if bw < min_side or bh < min_side:
        return None
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    c = img[y1:y2, x1:x2]
    return c.copy() if c.size else None


class Verifier:
    def __init__(self, analyzer, cache_file: Optional[Path] = None):
        self.a, self.cache_file = analyzer, cache_file
        self._t: dict[str, np.ndarray] = {}
        if cache_file and cache_file.exists():
            try:
                z = np.load(cache_file, allow_pickle=False)
                self._t = dict(zip(z["labels"].tolist(), z["vecs"]))
            except Exception:  # noqa: BLE001
                self._t = {}

    def _texts(self, labels: list[str]) -> np.ndarray:
        miss = [l for l in labels if l not in self._t]
        if miss:
            v = self.a.embed_texts([f"a photo of a {l}" for l in miss])
            for l, x in zip(miss, v):
                self._t[l] = x
            if self.cache_file:
                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(self.cache_file, labels=np.array(list(self._t)), vecs=np.stack(list(self._t.values())))
                except Exception:  # noqa: BLE001
                    pass
        return np.stack([self._t[l] for l in labels])

    def verify(self, crops: list, labels: list[str], detector_labels: list[str]) -> list[Optional[tuple[str, float]]]:
        """One entry per crop: (name, probability) or None when the box should be dropped."""
        if not crops:
            return []
        names = list(dict.fromkeys(labels + [d for d in detector_labels if d]))
        allv = names + NOT_AN_OBJECT
        T = self._texts(allv)
        F = self.a.embed_images(crops)
        out: list[Optional[tuple[str, float]]] = []
        for f, dl in zip(F, detector_labels):
            z = 100.0 * (T @ f)
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            order = np.argsort(-p)
            k = int(order[0])
            if k >= len(names):                       # CLIP says: not an object
                out.append(None)
                continue
            top3 = [names[int(j)] for j in order[:3] if int(j) < len(names)]
            pdet = float(p[names.index(dl)]) if dl in names else 0.0
            if dl and dl in top3 and pdet >= AGREE and pdet >= 0.5 * float(p[k]):
                out.append((dl, max(pdet, float(p[k]) * 0.6)))        # detector and CLIP agree closely: keep the detector's name
            elif p[k] >= ACCEPT and (not dl or p[k] >= 2.0 * max(pdet, 1e-6) or p[k] >= 0.75):
                out.append((names[k], float(p[k])))                   # CLIP renames only when it is clearly sure
            else:
                out.append(None)
        return out


_V: Optional[Verifier] = None


def get_verifier(models_dir: Path):
    """None while the CLIP model is still being prepared."""
    global _V
    from attributes import get_analyzer
    a = get_analyzer()
    if not a.is_ready():
        a.load_async()
        return None
    if _V is None or _V.a is not a:
        _V = Verifier(a, models_dir / "clip_text_cache.npz")
    return _V


# ───────────── general sanity checks that do not depend on the picture ─────────────
# A detector proposes boxes on anything round or bright (bokeh light circles -> "tennis ball", a smooth patch -> "flower").
# Real objects have a crisp outline somewhere; out-of-focus blobs do not. We measure how crisp the strongest edges
# inside a box are, compared with the sharpest edges in the SAME picture, so a soft photo is not punished as a whole.
def gradient_map(img: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def edge_ref(grad: np.ndarray) -> float:
    return float(np.percentile(grad, 99.5)) + 1e-6


def crispness(grad: np.ndarray, ref: float, box) -> float:
    """0..1+ : strength of the box's strongest edges relative to the sharpest edges of the whole picture."""
    h, w = grad.shape
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    return float(np.percentile(grad[y1:y2, x1:x2], 98)) / ref


BLUR_BELOW = float(os.environ.get("RCV_BLUR_BELOW", "0.22"))   # crisper than this = a real, in-focus thing
AUTO_FLOOR = float(os.environ.get("RCV_AUTO_FLOOR", "0.40"))   # Auto mode never shows names this unsure
