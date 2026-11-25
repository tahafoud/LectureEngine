"""أدوات تشابه الإطارات (SSIM) والحدّة وقواعد الوسم الأولي السريعة."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils import imread

THUMB = (320, 180)
_C1, _C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2


def thumb_of(img) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    k = (7, 7)
    mu1, mu2 = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    s11 = cv2.GaussianBlur(a * a, k, 1.5) - mu1 * mu1
    s22 = cv2.GaussianBlur(b * b, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(a * b, k, 1.5) - mu1 * mu2
    num = (2 * mu1 * mu2 + _C1) * (2 * s12 + _C2)
    den = (mu1 * mu1 + mu2 * mu2 + _C1) * (s11 + s22 + _C2)
    return float((num / den).mean())


def diff_frac(a: np.ndarray, b: np.ndarray, thr: float = 30.0) -> float:
    """نسبة البكسلات التي تغيّرت بوضوح بين إطارين (0..1).

    أدقّ من SSIM العام لشرائح بيضاء قليلة النص: المؤشر ≈ 0.0002 بينما شريحة جديدة بنص حقيقي ≥ 0.01.
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    a, b = cv2.GaussianBlur(a, (3, 3), 0), cv2.GaussianBlur(b, (3, 3), 0)
    return float((np.abs(a - b) > thr).mean())


def sharpness(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    if w > 640:
        g = cv2.resize(g, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def is_blank_stats(mean: float, std: float) -> bool:
    """شاشة سوداء أو بيضاء فارغة."""
    return (mean < 25 and std < 10) or (mean > 235 and std < 4)


def _feat(path):
    img = imread(path)
    if img is None:
        return None
    th = thumb_of(img)
    return {"thumb": th, "mean": float(th.mean()), "std": float(th.std()), "lap": sharpness(img)}


def prelabel(paths: list) -> list:
    """وسم أولي بقواعد خفيفة: slide | blank | transition (مرتّب زمنياً)."""
    feats = [_feat(p) for p in paths]
    laps = [f["lap"] for f in feats if f and not is_blank_stats(f["mean"], f["std"])]
    lap_med = float(np.median(laps)) if laps else 1.0
    labels = []
    for i, f in enumerate(feats):
        if f is None:
            labels.append("transition")
            continue
        if is_blank_stats(f["mean"], f["std"]):
            labels.append("blank")
            continue
        sp = ssim(feats[i - 1]["thumb"], f["thumb"]) if i > 0 and feats[i - 1] else 1.0
        sn = ssim(f["thumb"], feats[i + 1]["thumb"]) if i + 1 < len(feats) and feats[i + 1] else 1.0
        blurry = f["lap"] < 0.25 * lap_med
        isolated = sp < 0.45 and sn < 0.45 and f["lap"] < 0.6 * lap_med
        labels.append("transition" if (blurry or isolated) else "slide")
    return labels
