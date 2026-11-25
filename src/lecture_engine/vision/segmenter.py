"""تحويل تسميات الإطارات إلى شرائح بنطاق زمني (start_time → end_time) وصورة ممثِّلة."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..utils import imread
from .similarity import diff_frac, sharpness, thumb_of


def build_slides(samples: list, interval: float, duration: float, change_thr: float = 0.004,
                 min_frames: int = 2, max_gap: float = 60.0) -> list:
    """samples: [{"t","path","label"}] مرتبة زمنياً.

    إطار جديد يُضم لنفس الشريحة إذا تغيّر أقل من change_thr من مساحته عن آخر إطار شريحة وكان الفاصل ≤ max_gap ثانية؛
    فالتأشير/الكتابة الخفيفة لا تقسم الشريحة، وعودة نفس الشريحة بعد لقطة انتقال تُدمج.
    """
    segs, cur, prev_th, prev_t = [], None, None, None
    for s in samples:
        if s["label"] != "slide":
            continue
        img = imread(s["path"])
        if img is None:
            continue
        th = thumb_of(img)
        same = (cur is not None and prev_th is not None and (s["t"] - prev_t) <= max_gap
                and diff_frac(prev_th, th) <= change_thr)
        if same:
            cur.append(s)
        else:
            cur = [s]
            segs.append(cur)
        prev_th, prev_t = th, s["t"]

    kept = [g for g in segs if len(g) >= min_frames]
    if not kept:
        kept = segs          # فيديو قصير/شرائح سريعة: لا نحذف كل شيء
    slides = []
    for g in kept:
        slides.append({"start_time": g[0]["t"], "frames": g})
    for i, sl in enumerate(slides):
        nxt = slides[i + 1]["start_time"] if i + 1 < len(slides) else None
        last_t = sl["frames"][-1]["t"] + interval
        sl["end_time"] = round(nxt if nxt is not None else max(duration, last_t), 3)
        sl["rep_path"] = str(_pick_rep(sl["frames"]))
        sl["n_frames"] = len(sl["frames"])
    return slides


def _pick_rep(frames: list) -> Path:
    """أوضح إطار من النصف الأخير للشريحة (يكون المحتوى قد اكتمل)."""
    cand = frames[len(frames) // 2:] or frames
    if len(cand) > 6:
        step = len(cand) / 6
        cand = [cand[int(i * step)] for i in range(6)]
    best, best_s = cand[-1], -1.0
    for f in cand:
        img = imread(f["path"])
        if img is None:
            continue
        sc = sharpness(img)
        if sc > best_s:
            best, best_s = f, sc
    return Path(best["path"])


def save_slides(slides: list, slides_dir: Path) -> list:
    """ينسخ الصور الممثِّلة إلى slides/slide_001.jpg ويعيد سجلات JSON خفيفة."""
    slides_dir = Path(slides_dir)
    slides_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, s in enumerate(slides, 1):
        name = f"slide_{i:03d}.jpg"
        shutil.copy2(s["rep_path"], slides_dir / name)
        out.append({"slide_id": i, "start_time": s["start_time"], "end_time": s["end_time"],
                    "image": f"slides/{name}", "n_frames": s["n_frames"], "ocr_text": ""})
    return out
