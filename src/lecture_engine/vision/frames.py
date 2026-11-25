"""استخراج إطارات بفاصل زمني منتظم (افتراضياً كل ثانيتين) إلى مجلد."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import cv2

from ..utils import imwrite


def video_duration(path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return float(n / fps) if fps > 0 and n > 0 else 0.0
    finally:
        cap.release()


def extract_frames(video, out_dir, interval: float = 2.0, max_side: int = 1280,
                   quality: int = 90, on_progress: Optional[Callable[[float], None]] = None) -> list:
    """يعيد قائمة [(t_seconds, Path)] مرتبة زمنياً. الاسم: f_<milliseconds>.jpg"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"تعذر فتح الفيديو: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval)))
    items, idx, last_report = [], 0, 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    if max(h, w) > max_side:
                        s = max_side / max(h, w)
                        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                    t = idx / fps
                    p = out_dir / f"f_{int(round(t * 1000)):08d}.jpg"
                    if imwrite(p, frame, quality):
                        items.append((round(t, 3), p))
            idx += 1
            if on_progress and total and idx - last_report > 500:
                last_report = idx
                on_progress(min(1.0, idx / total))
    finally:
        cap.release()
    if not items:
        raise RuntimeError("لم يُستخرج أي إطار؛ الفيديو تالف أو صيغته غير مدعومة.")
    return items
