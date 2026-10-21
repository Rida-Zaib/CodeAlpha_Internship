"""
Traffic Survey Report - turns the cleaned tracks of one video into the numbers a traffic survey needs:

  * how many vehicles / pedestrians of each kind crossed the line, in which direction, in each time slot,
  * the busiest slot (peak),
  * the vehicle types and models seen (tempo, auto rickshaw, Maruti Swift ...),
  * flags: vehicles that stood still for a long time (parked / blocked lane) and vehicles going against the expected direction.

Everything is computed from the final tracks (after duplicates and flicker were removed), so it is the same data as the
video and the CSV. Counts are automatic estimates - check a sample against a manual count before relying on them.
"""
from __future__ import annotations

import base64
import csv
import html
import io
import math
from collections import Counter
from pathlib import Path
from typing import Optional

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}
LABEL = {"person": "Pedestrians", "car": "Cars", "truck": "Trucks / tempos", "bus": "Buses", "motorcycle": "Motorcycles / scooters",
         "bicycle": "Bicycles"}


def auto_bin_seconds(duration: float) -> int:
    return 10 if duration < 60 else 60 if duration < 600 else 300 if duration < 3600 else 900


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


def _type_of(attr: str) -> Optional[str]:
    """'Maruti Suzuki Swift (hatchback)' -> ('Maruti Suzuki Swift', 'hatchback'); 'tempo' -> (None, 'tempo')."""
    a = (attr or "").strip()
    return a or None


def build_report(timeline: list, fps: float, W: int, H: int, line_pct: int, bin_seconds: int = 0, expected: str = "",
                 stop_seconds: float = 10.0, n_frames: int = 0, notes: Optional[list] = None) -> dict:
    """timeline rows: (frame, id, cls, cx, cy, crossed, direction, attrs); frame is 1-based."""
    fps = fps if fps > 0 else 25.0
    n_frames = n_frames or (max((r[0] for r in timeline), default=0))
    duration = n_frames / fps
    bin_s = int(bin_seconds) if bin_seconds and bin_seconds > 0 else auto_bin_seconds(duration)
    n_bins = max(1, int(math.ceil(duration / bin_s)))
    diag = math.hypot(W, H)

    tracks: dict[int, dict] = {}
    for f, tid, cls, cx, cy, crossed, direction, attrs in timeline:
        t = tracks.setdefault(tid, {"id": tid, "cls": cls, "pts": [], "first": f, "attr": "", "cross": []})
        t["pts"].append((f, cx, cy))
        t["cls"] = cls
        if attrs and cls in VEHICLES:
            t["attr"] = attrs
        if crossed and direction:
            t["cross"].append((f, direction))

    def slot(frame: int) -> int:
        return min(n_bins - 1, int((frame - 1) / fps // bin_s))

    by_class: dict[str, dict] = {}
    bins = [{"index": i, "label": f"{_fmt(i * bin_s)} - {_fmt(min(duration, (i + 1) * bin_s))}", "arrivals": 0, "crossings": 0, "by_class": Counter()}
            for i in range(n_bins)]
    types: Counter = Counter()
    headings: Counter = Counter()
    stopped: list[dict] = []
    wrong: list[dict] = []
    k = max(1, int(round(fps)))
    still_thr = 0.015 * diag
    move_thr = 0.12

    for t in tracks.values():
        cls, pts = t["cls"], t["pts"]
        c = by_class.setdefault(cls, {"cls": cls, "name": LABEL.get(cls, cls.title()), "down": 0, "up": 0, "total": 0, "unique": 0})
        c["unique"] += 1
        bins[slot(t["first"])]["arrivals"] += 1
        seen = set()
        for f, d in t["cross"]:
            if d in seen:
                continue
            seen.add(d)
            c[d] = c.get(d, 0) + 1
            c["total"] += 1
            bins[slot(f)]["crossings"] += 1
            bins[slot(f)]["by_class"][cls] += 1
        if cls in VEHICLES and t["attr"]:
            types[t["attr"]] += 1
        if cls not in VEHICLES:
            continue
        dx, dy = pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]
        if math.hypot(dx, dy) < 0.05 * diag:
            headings["standing / short"] += 1
        else:
            headings["down" if abs(dy) >= abs(dx) and dy > 0 else "up" if abs(dy) >= abs(dx) else "right" if dx > 0 else "left"] += 1
        # longest time standing still (compare with where it was one second earlier)
        run = best = 0
        best_start = pts[0][0]
        run_start = pts[0][0]
        for i in range(k, len(pts)):
            if math.hypot(pts[i][1] - pts[i - k][1], pts[i][2] - pts[i - k][2]) < still_thr:
                if run == 0:
                    run_start = pts[i - k][0]
                run += 1
                if run > best:
                    best, best_start = run, run_start
            else:
                run = 0
        secs = (best + k) / fps if best else 0.0
        if secs >= stop_seconds:
            stopped.append({"id": t["id"], "cls": cls, "label": t["attr"] or LABEL.get(cls, cls), "start": _fmt((best_start - 1) / fps),
                            "seconds": round(secs, 1)})
        if expected in ("up", "down", "left", "right"):
            along = {"down": dy, "up": -dy, "right": dx, "left": -dx}[expected]
            span = H if expected in ("up", "down") else W
            if along < -move_thr * span:
                wrong.append({"id": t["id"], "cls": cls, "label": t["attr"] or LABEL.get(cls, cls), "first": _fmt((t["first"] - 1) / fps)})

    veh_unique = sum(c["unique"] for k_, c in by_class.items() if k_ in VEHICLES)
    ped_unique = by_class.get("person", {}).get("unique", 0)
    peak = max(bins, key=lambda b: (b["arrivals"], b["crossings"])) if bins else None
    order = sorted(by_class.values(), key=lambda c: (-(c["cls"] in VEHICLES), -c["unique"]))
    return {
        "duration_s": round(duration, 1), "fps": round(fps, 1), "frames": n_frames, "bin_seconds": bin_s, "line_pct": line_pct,
        "totals": {"vehicles": veh_unique, "pedestrians": ped_unique, "crossings": sum(c["total"] for c in by_class.values())},
        "by_class": order,
        "bins": [{"label": b["label"], "arrivals": b["arrivals"], "crossings": b["crossings"], "by_class": dict(b["by_class"])} for b in bins],
        "peak": ({"label": peak["label"], "arrivals": peak["arrivals"], "crossings": peak["crossings"]} if peak and peak["arrivals"] else None),
        "types": [{"label": l, "count": n} for l, n in types.most_common(25)],
        "headings": dict(headings),
        "stopped": sorted(stopped, key=lambda s: -s["seconds"])[:50],
        "wrong_way": wrong[:50],
        "settings": {"expected_direction": expected or "not set", "stop_seconds": stop_seconds},
        "notes": list(notes or []),
    }


def to_csv(rep: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["section", "label", "down", "up", "total_crossings", "unique_objects"])
    for c in rep["by_class"]:
        w.writerow(["class", c["name"], c.get("down", 0), c.get("up", 0), c["total"], c["unique"]])
    w.writerow([])
    w.writerow(["time_slot", "arrivals", "line_crossings"])
    for b in rep["bins"]:
        w.writerow([b["label"], b["arrivals"], b["crossings"]])
    w.writerow([])
    w.writerow(["vehicle_type_or_model", "count"])
    for t in rep["types"]:
        w.writerow([t["label"], t["count"]])
    w.writerow([])
    w.writerow(["flag", "track_id", "kind", "detail"])
    for s in rep["stopped"]:
        w.writerow(["stopped", s["id"], s["label"], f"from {s['start']} for {s['seconds']} s"])
    for s in rep["wrong_way"]:
        w.writerow(["wrong_way", s["id"], s["label"], f"first seen {s['first']}"])
    return out.getvalue()


def to_html(rep: dict, heatmap: Optional[Path] = None, title: str = "Traffic Survey Report") -> str:
    e = html.escape
    img = ""
    if heatmap and Path(heatmap).exists():
        img = f'<h2>Where traffic moved and stopped</h2><img alt="heatmap" src="data:image/png;base64,{base64.b64encode(Path(heatmap).read_bytes()).decode()}">'
    mx = max([b["arrivals"] for b in rep["bins"]] + [1])
    cls_rows = "".join(f"<tr><td>{e(c['name'])}</td><td>{c.get('down', 0)}</td><td>{c.get('up', 0)}</td><td><b>{c['total']}</b></td><td>{c['unique']}</td></tr>"
                       for c in rep["by_class"]) or "<tr><td colspan=5>Nothing was detected.</td></tr>"
    bin_rows = "".join(f"<tr><td>{e(b['label'])}</td><td>{b['arrivals']}</td><td>{b['crossings']}</td>"
                       f"<td><div class='bar' style='width:{int(100 * b['arrivals'] / mx)}%'></div></td></tr>" for b in rep["bins"])
    type_rows = "".join(f"<tr><td>{e(t['label'])}</td><td>{t['count']}</td></tr>" for t in rep["types"]) or "<tr><td colspan=2>No vehicle types identified (turn on 'Identify vehicle types &amp; models').</td></tr>"
    stop_rows = "".join(f"<tr><td>ID {s['id']}</td><td>{e(s['label'])}</td><td>{s['start']}</td><td>{s['seconds']} s</td></tr>" for s in rep["stopped"]) or "<tr><td colspan=4>None.</td></tr>"
    wr_rows = "".join(f"<tr><td>ID {s['id']}</td><td>{e(s['label'])}</td><td>{s['first']}</td></tr>" for s in rep["wrong_way"]) or \
        ("<tr><td colspan=3>None.</td></tr>" if rep["settings"]["expected_direction"] != "not set" else "<tr><td colspan=3>Not checked (no expected direction was set).</td></tr>")
    peak = rep["peak"]
    peak_txt = f"{e(peak['label'])} - {peak['arrivals']} new objects, {peak['crossings']} line crossings" if peak else "n/a"
    notes = "".join(f"<li>{e(n)}</li>" for n in rep["notes"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><style>
body{{font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#10262b}}
h1{{margin:0 0 4px}}h2{{margin:28px 0 8px;border-bottom:2px solid #c2a25a;padding-bottom:4px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}.card{{flex:1 1 150px;border:1px solid #c9d6d8;border-radius:10px;padding:10px 14px}}
.card b{{display:block;font-size:1.6rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5dfe0;padding:6px 10px;text-align:left}}
th{{background:#eef4f5}}.bar{{height:12px;background:#1f7a86;border-radius:6px}}img{{max-width:100%;border-radius:8px}}.muted{{color:#5a6f73;font-size:.88rem}}
@media print{{body{{margin:0}}}}</style></head><body>
<h1>{e(title)}</h1>
<div class="muted">Video length {_fmt(rep['duration_s'])} &middot; {rep['frames']} frames at {rep['fps']} fps &middot; counting line at {rep['line_pct']}% of the picture height &middot; time slots of {rep['bin_seconds']} s</div>
<div class="cards"><div class="card">Vehicles<b>{rep['totals']['vehicles']}</b></div><div class="card">Pedestrians<b>{rep['totals']['pedestrians']}</b></div>
<div class="card">Line crossings<b>{rep['totals']['crossings']}</b></div><div class="card">Busiest slot<b style="font-size:1rem">{peak_txt}</b></div></div>
<h2>Counts by kind and direction</h2><table><tr><th>Kind</th><th>Down</th><th>Up</th><th>Crossed the line</th><th>Seen in total</th></tr>{cls_rows}</table>
<p class="muted">Down / Up = direction on screen. "Seen in total" counts every separate object in the video, also those that never reached the line.</p>
<h2>By time slot</h2><table><tr><th>Slot</th><th>New objects</th><th>Line crossings</th><th>Busy-ness</th></tr>{bin_rows}</table>
<h2>Vehicle types and models</h2><table><tr><th>Type / model (estimated from appearance)</th><th>Count</th></tr>{type_rows}</table>
<h2>Vehicles that stood still for {rep['settings']['stop_seconds']:g} s or more</h2><table><tr><th>Track</th><th>Vehicle</th><th>From</th><th>For</th></tr>{stop_rows}</table>
<h2>Going against the expected direction ({e(str(rep['settings']['expected_direction']))})</h2><table><tr><th>Track</th><th>Vehicle</th><th>First seen</th></tr>{wr_rows}</table>
{img}
<h2>How to read this</h2><ul><li>All numbers are automatic estimates from a computer-vision model. Before relying on them, count a few minutes by hand and compare.</li>
<li>Stopped vehicles include red-light and traffic-jam stops as well as parking; check the video at the time shown.</li>
<li>Vehicle models are guessed from appearance and are only shown when the match is clear.</li>{notes}</ul></body></html>"""
