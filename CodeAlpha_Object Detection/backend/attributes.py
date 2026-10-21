"""
People attributes from a crop of each detected person, using CLIP zero-shot classification:
person type (man / woman / boy / girl / teen / senior), bag, phone use and activity.

These are APPEARANCE-BASED ESTIMATES (CLIP is not a trained pedestrian-attribute model), so they can be wrong,
especially for small, blurry or partly hidden people. Weights (~600 MB, openai/clip-vit-base-patch32) are
downloaded once, the first time "Analyze people" is used.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

GENDER = ["a photo of a man", "a photo of a woman"]
AGE = ["a photo of a young child", "a photo of a teenager", "a photo of an adult", "a photo of an elderly person"]
AGE_KEYS = ["kid", "teen", "adult", "senior"]
BAG = ["a person carrying a backpack", "a person carrying a handbag or shoulder bag",
       "a person carrying a shopping bag", "a person with empty hands and no bag"]
BAG_KEYS = ["backpack", "handbag", "shopping bag", None]
PHONE = ["a person using a mobile phone", "a person not using a phone"]
ACT = ["a person reading a book", "a person writing with a pen on paper", "a person playing a game or a sport",
       "a person using a laptop or computer", "a person walking", "a person standing still", "a person sitting",
       "a person eating or drinking", "a person talking to someone", "a person running"]
ACT_KEYS = ["reading", "writing", "playing", "using laptop", "walking", "standing", "sitting", "eating/drinking", "talking", "running"]


def person_type(gender: str, age: str) -> str:
    g = gender == "female"
    return {"kid": "girl" if g else "boy", "teen": "teen girl" if g else "teen boy",
            "adult": "woman" if g else "man", "senior": "elderly woman" if g else "elderly man"}[age]


def make_tag(a: dict) -> str:
    parts = [a["type"]]
    if a.get("bag"):
        parts.append("with " + a["bag"])
    if a.get("phone"):
        parts.append("on phone")
    if a.get("activity"):
        parts.append(a["activity"])
    return " · ".join(parts)


def person_crop(img: np.ndarray, box, pad: float = 0.1) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 30:
        return None
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    c = img[y1:y2, x1:x2]
    return c if c.size else None


class PersonAnalyzer:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready = False
        self.error = None

    def is_ready(self) -> bool:
        return self.ready

    def ensure_loaded(self):
        with self.lock:
            if not self.ready:
                self._load()

    def load_async(self):
        """Start the (large, one-time) model download in the background without blocking the caller."""
        if self.ready or getattr(self, "_loading", False):
            return
        self._loading = True

        def run():
            try:
                self.ensure_loaded()
                self.error = None
            except Exception as e:  # noqa: BLE001
                self.error = str(e)[:160]
            finally:
                self._loading = False

        threading.Thread(target=run, daemon=True, name="rcv-people-load").start()

    def _load(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("'Analyze people' needs the transformers package: pip install transformers") from e
        name = os.environ.get("RCV_CLIP", "openai/clip-vit-base-patch32")
        self.torch = torch
        self.proc = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name).eval()
        self.text = {}
        for key, prompts in (("gender", GENDER), ("age", AGE), ("bag", BAG), ("phone", PHONE), ("act", ACT)):
            tok = self.proc(text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                f = self._tensor(self.model.get_text_features(**tok))
            self.text[key] = f / f.norm(dim=-1, keepdim=True)
        self.ready = True

    def _tensor(self, out):
        if isinstance(out, self.torch.Tensor):
            return out
        for k in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, k, None)
            if v is not None:
                return v
        return out[0]

    def analyze(self, crops: list) -> list[dict]:
        if not crops:
            return []
        with self.lock:
            if not self.ready:
                self._load()
            from PIL import Image
            torch = self.torch
            imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
            res = []
            for i in range(0, len(imgs), 16):
                inp = self.proc(images=imgs[i:i + 16], return_tensors="pt")
                with torch.no_grad():
                    f = self._tensor(self.model.get_image_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                P = {k: (100.0 * f @ t.T).softmax(dim=-1).cpu().numpy() for k, t in self.text.items()}
                for j in range(len(f)):
                    res.append(self._decide(P["gender"][j], P["age"][j], P["bag"][j], P["phone"][j], P["act"][j]))
            return res

    def embed_images(self, crops: list) -> np.ndarray:
        """L2-normalised CLIP image embeddings (n, d) for BGR crops. Shared with the vehicle recogniser."""
        with self.lock:
            if not self.ready:
                self._load()
            from PIL import Image
            torch = self.torch
            out = []
            for i in range(0, len(crops), 16):
                imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + 16]]
                inp = self.proc(images=imgs, return_tensors="pt")
                with torch.no_grad():
                    f = self._tensor(self.model.get_image_features(**inp))
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            return np.vstack(out) if out else np.zeros((0, 512), dtype=np.float32)

    def embed_texts(self, prompts: list) -> np.ndarray:
        """L2-normalised CLIP text embeddings (n, d)."""
        with self.lock:
            if not self.ready:
                self._load()
            torch = self.torch
            out = []
            for i in range(0, len(prompts), 64):
                tok = self.proc(text=prompts[i:i + 64], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    f = self._tensor(self.model.get_text_features(**tok))
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            return np.vstack(out) if out else np.zeros((0, 512), dtype=np.float32)

    @staticmethod
    def _decide(g, age, bag, phone, act) -> dict:
        gender = "female" if g[1] > g[0] else "male"
        age_k = AGE_KEYS[int(np.argmax(age))]
        bag_label = None
        if float(np.sum(bag[:3])) > 0.5:
            bag_label = BAG_KEYS[int(np.argmax(bag[:3]))]
        a_i = int(np.argmax(act))
        a = {"gender": gender, "age": age_k, "type": person_type(gender, age_k), "bag": bag_label,
             "phone": bool(phone[0] > 0.5), "activity": ACT_KEYS[a_i] if float(act[a_i]) > 0.35 else None}
        a["tag"] = make_tag(a)
        return a


class FakeAnalyzer:
    """Deterministic stand-in used by the offline tests (RCV_FAKE_DETECTOR=1)."""

    ready = True
    error = None

    def is_ready(self) -> bool:
        return True

    def ensure_loaded(self):
        pass

    def load_async(self):
        pass

    def embed_texts(self, prompts: list) -> np.ndarray:
        import zlib
        v = np.stack([np.random.default_rng(zlib.crc32(p.encode())).normal(size=64) for p in prompts]) if prompts else np.zeros((0, 64))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32) if len(v) else v.astype(np.float32)

    def embed_images(self, crops: list) -> np.ndarray:
        """Bright crops look like a tempo, dark ones like a hatchback (lets the tests exercise the whole vehicle path)."""
        return self.embed_texts(["a photo of a tempo" if float(c.mean()) >= 128 else "a photo of a hatchback" for c in crops])

    def analyze(self, crops: list) -> list[dict]:
        out = []
        for c in crops:
            k = int(c.mean()) % 4
            g, ag = ("female", "adult") if k == 0 else ("male", "adult") if k == 1 else ("female", "kid") if k == 2 else ("male", "kid")
            a = {"gender": g, "age": ag, "type": person_type(g, ag), "bag": "backpack" if k in (0, 2) else None,
                 "phone": k == 0, "activity": ["reading", "writing", "playing", "walking"][k]}
            a["tag"] = make_tag(a)
            out.append(a)
        return out


_ANALYZER = None


def get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = FakeAnalyzer() if os.environ.get("RCV_FAKE_DETECTOR") == "1" else PersonAnalyzer()
    return _ANALYZER


def summarize(attrs: list[dict]) -> dict:
    """Counts for the dashboard (only non-zero entries)."""
    s: dict = {}

    def add(k, n=1):
        s[k] = s.get(k, 0) + n

    for a in attrs:
        t = a["type"]
        if t == "woman" or t == "elderly woman":
            add("Women")
        elif t == "man" or t == "elderly man":
            add("Men")
        elif t == "girl":
            add("Girls")
        elif t == "boy":
            add("Boys")
        else:
            add("Teens")
        if a["age"] == "senior":
            add("Seniors")
        if a.get("bag"):
            add("With bags")
        if a.get("phone"):
            add("On phone")
        if a.get("activity") in ("reading", "writing", "playing", "using laptop"):
            add(a["activity"].capitalize())
    return s
