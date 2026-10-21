"""Duplicate / false-detection control and vehicle recognition. Run: RCV_FAKE_DETECTOR=1 pytest -q"""
import os, sys
os.environ["RCV_FAKE_DETECTOR"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pathlib

import numpy as np

import engine
import vehicles

NAMES = {0: "person", 1: "bicycle", 2: "car", 7: "truck"}
HTML = (pathlib.Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")


def row(box, tid, conf, cls):
    return [*box, tid, conf, cls]


def test_one_car_called_car_and_truck_becomes_one_object():
    """The reported bug: the white car was tracked twice (truck 0.51 + car 0.42) for the whole clip."""
    car = (400, 300, 520, 420)
    raw = [np.array([row(car, 2, .51, 7), row(car, 4, .42, 2)], float) for _ in range(30)]
    frames, info = engine.consolidate(raw, NAMES, 25.0, .35)
    assert info["merged"] == 1 and info["kept"] == 1
    assert all(len(f) == 1 for f in frames)                       # one box per frame, not two
    assert {t["cls"] for f in frames for t in f} == {"car"}       # one stable class for its whole life (car beats a hesitant truck)


def test_flicker_and_weak_tracks_are_dropped():
    real = (100, 100, 160, 260)
    raw = []
    for k in range(40):
        rows = [row(real, 1, .8, 0)]
        if k in (10, 11):                                         # a 2-frame ghost
            rows.append(row((500, 50, 560, 120), 9, .9, 7))
        rows.append(row((300, 300, 340, 340), 5, .36, 2))         # always weak: never clearly confident, short-lived
        raw.append(np.array(rows, float))
    raw = raw[:12]                                                # weak track only 12 frames (< 4 x min_len)
    frames, info = engine.consolidate(raw, NAMES, 25.0, .35)
    assert info["kept"] == 1 and info["removed_short"] == 2
    assert {t["cls"] for f in frames for t in f} == {"person"}


def test_class_does_not_flip_frame_to_frame():
    b = (50, 50, 200, 200)
    raw = [np.array([row(b, 3, .6 if k % 3 else .4, 7 if k % 3 else 2)], float) for k in range(30)]
    frames, _ = engine.consolidate(raw, NAMES, 25.0, .35)
    assert len({t["cls"] for f in frames for t in f}) == 1


def test_ids_are_clean_and_in_order():
    raw = [np.array([row((10, 10, 60, 100), 17, .8, 0)], float) for _ in range(10)]
    raw += [np.array([row((10, 10, 60, 100), 17, .8, 0), row((300, 10, 360, 100), 42, .8, 0)], float) for _ in range(10)]
    frames, _ = engine.consolidate(raw, NAMES, 25.0, .35)
    assert sorted({t["id"] for f in frames for t in f}) == [1, 2]


def test_person_on_bicycle_is_not_merged():
    raw = [np.array([row((100, 100, 160, 260), 1, .8, 0), row((80, 160, 190, 290), 2, .8, 1)], float) for _ in range(20)]
    frames, info = engine.consolidate(raw, NAMES, 25.0, .35)
    assert info["kept"] == 2 and {t["cls"] for t in frames[0]} == {"person", "bicycle"}


def test_image_filter_removes_same_family_double():
    d = np.array([[100, 100, 300, 250, .51, 7], [102, 101, 298, 249, .42, 2], [400, 100, 500, 300, .8, 0]], float)
    out = engine.filter_dets(d, NAMES, 800, 600)
    assert len(out) == 2 and set(out[:, 5].astype(int)) == {7, 0}


class _Stub:
    """Embeddings where each label is an axis: lets us check the decision logic without CLIP."""
    def __init__(self, truth):
        self.truth, self.axes = truth, {}

    def _vec(self, label):
        i = self.axes.setdefault(label, len(self.axes))
        v = np.zeros(2048, np.float32)
        v[i] = 1.0
        return v

    def embed_texts(self, prompts):
        return np.stack([self._vec(p.replace("a photo of a ", "").replace(", a vehicle", "")) for p in prompts])

    def embed_images(self, crops):
        return np.stack([self._vec(self.truth[int(c.flat[0])]) for c in crops])


def test_vehicle_type_model_and_class_correction():
    a = _Stub({1: "tempo", 2: "hatchback", 3: "tata ace", 4: "maruti suzuki swift"})
    clf = vehicles.VehicleClassifier(a)
    crop = lambda k: np.full((40, 40, 3), k, np.uint8)
    res = clf.classify({1: [crop(1)], 2: [crop(2)]}, {1: "car", 2: "car"}, extra=["tata ace"])
    assert res[1]["type"] == "tempo" and res[1]["class_override"] == "truck"   # YOLO said car, CLIP is sure it is a tempo (light truck)
    assert res[2]["type"] == "hatchback" and res[2]["class_override"] is None
    res = clf.classify({3: [crop(1)]}, {3: "car"}, extra=["tata ace"])
    assert res[3]["label"] in ("tempo", "tata ace (tempo)")
    # car wrongly called truck: CLIP is sure it is a hatchback -> class corrected to car
    res = clf.classify({7: [crop(2)]}, {7: "truck"})
    assert res[7]["class_override"] == "car"


def test_free_format_user_name_is_matched():
    a = _Stub({5: "mahindra bolero pickup"})
    clf = vehicles.VehicleClassifier(a)
    res = clf.classify({1: [np.full((40, 40, 3), 5, np.uint8)]}, {1: "truck"}, extra=["mahindra bolero pickup"])
    assert res[1]["model"] == "mahindra bolero pickup"


def test_catalogue_is_large_and_covers_the_requested_kinds():
    n = sum(len(v) for v in vehicles.MODELS.values())
    assert n > 900
    for needle in ("Tata Ace", "Bajaj Maxima", "Royal Enfield Classic 350", "Maruti Suzuki Swift", "Hero Splendor"):
        assert any(needle.lower() in m.lower() for ms in vehicles.MODELS.values() for m in ms), needle
    for t in ("tempo", "auto rickshaw", "dump truck", "mountain bike", "cruiser motorcycle", "double decker bus"):
        assert any(t in ts for ts in vehicles.TYPES.values()), t


def test_vocabulary_is_in_the_thousands():
    import vocab
    assert len(vocab.UNIVERSAL_VOCAB) > 2000 and "lion" in vocab.UNIVERSAL_VOCAB and "whiteboard" in vocab.UNIVERSAL_VOCAB


def test_result_video_plays_once():
    assert "el.loop = false" in HTML and "el.autoplay = false" in HTML and "el.loop = true" not in HTML


# ---- CLIP second check for open-vocabulary detections ----
import verify


class _VStub(_Stub):
    def embed_images(self, crops):
        return np.stack([self._vec(self.truth[int(c.flat[0])]) for c in crops])


def test_verifier_renames_drops_and_keeps():
    a = _VStub({1: "squirrel", 2: "background", 3: "dragonfly"})
    v = verify.Verifier(a)
    crop = lambda k: np.full((40, 40, 3), k, np.uint8)
    labels = ["squirrel", "beaver", "dragonfly", "flamingo", "mushroom"]
    out = v.verify([crop(1), crop(2), crop(3)], labels, ["beaver", "tree trunk", "flamingo"])
    assert out[0][0] == "squirrel"         # detector said beaver, CLIP knows it is a squirrel
    assert out[1] is None                  # "not an object" -> box removed
    assert out[2][0] == "dragonfly"        # detector said flamingo


def test_auto_vocab_has_no_scenery_and_menus_keep_it():
    import vocab
    for bad in ("jungle", "tree trunk", "stone wall", "forest", "sky"):
        assert bad not in vocab.DETECT_SET and bad in vocab.UNIVERSAL_VOCAB, bad
    for good in ("squirrel", "dragonfly", "mushroom", "flamingo", "lion", "chickadee"):
        assert good in vocab.DETECT_SET, good


# ---- Traffic Survey Report ----
import report as R


def _tl(tid, cls, pts, crossed_at=None, direction=None, attr=""):
    out = []
    for n, (x, y) in enumerate(pts, start=1):
        c = crossed_at == n
        out.append((n, tid, cls, x, y, c, direction if c else None, attr))
    return out


def test_report_counts_directions_bins_types_and_flags():
    fps, W, H = 10.0, 1000, 1000
    tl = []
    tl += _tl(1, "car", [(100, 100 + 20 * i) for i in range(40)], 25, "down", "Maruti Suzuki Swift (hatchback)")          # moves down
    tl += _tl(2, "car", [(300, 900 - 20 * i) for i in range(40)], 20, "up")                                              # moves UP (wrong way if expected=down)
    tl += [(i + 30, 3, "truck", 600, 500, False, None, "tempo") for i in range(1, 201)]                                 # parked 20 s
    tl += _tl(4, "person", [(800, 100 + 5 * i) for i in range(30)])
    rep = R.build_report(sorted(tl), fps, W, H, 50, bin_seconds=10, expected="down", stop_seconds=10, n_frames=230)
    cars = next(c for c in rep["by_class"] if c["cls"] == "car")
    assert cars["down"] == 1 and cars["up"] == 1 and cars["total"] == 2 and cars["unique"] == 2
    assert rep["totals"]["vehicles"] == 3 and rep["totals"]["pedestrians"] == 1
    assert [w["id"] for w in rep["wrong_way"]] == [2]
    assert [s["id"] for s in rep["stopped"]] == [3] and rep["stopped"][0]["seconds"] >= 19
    assert {"label": "tempo", "count": 1} in rep["types"]
    assert len(rep["bins"]) == 3 and rep["peak"]["arrivals"] >= 2
    html = R.to_html(rep)
    assert "Maruti Suzuki Swift" in html and "Traffic Survey Report" in html
    assert "down" in R.to_csv(rep) and "stopped" in R.to_csv(rep)


def test_report_auto_bin_and_empty_video():
    assert R.auto_bin_seconds(30) == 10 and R.auto_bin_seconds(300) == 60 and R.auto_bin_seconds(7200) == 900
    rep = R.build_report([], 25.0, 640, 360, 50, n_frames=100)
    assert rep["totals"]["vehicles"] == 0 and rep["peak"] is None and "Nothing was detected" in R.to_html(rep)


def test_video_job_writes_report(tmp_path):
    import cv2
    w = cv2.VideoWriter(str(tmp_path / "in.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(40):
        w.write(np.full((240, 320, 3), 60, np.uint8))
    w.release()
    job = dict(max_frames=0, model="yolov8n.pt", tracker="sort", conf=.35, classes=["person", "car"], line=50, report_bin=2, expected_dir="down", stop_seconds=5)
    engine.run_video_job(job, tmp_path / "in.mp4", tmp_path, lambda p: None)
    assert (tmp_path / "report.html").exists() and (tmp_path / "report.csv").exists()
    assert job["stats"]["report"]["bin_seconds"] == 2 and job["stats"]["report"]["frames"] == 40
