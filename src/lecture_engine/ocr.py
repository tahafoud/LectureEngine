"""قراءة نص الشرائح بـ PaddleOCR (يدعم إصدارات 2.x و3.x) مع حماية كاملة من الأخطاء."""
from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

from .utils import imread, log

_AR = re.compile(r"[؀-ۿ]")
_LAT = re.compile(r"[A-Za-z]")


def enhance_for_ocr(img: np.ndarray, target_w: int = 1600) -> np.ndarray:
    """تكبير الصور الصغيرة + تحسين تباين خفيف (CLAHE على قناة الإضاءة)."""
    h, w = img.shape[:2]
    if w < target_w:
        s = target_w / w
        img = cv2.resize(img, (target_w, int(h * s)), interpolation=cv2.INTER_CUBIC)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:  # noqa: BLE001
        pass
    return img


def _make_paddle(lang: str):
    from paddleocr import PaddleOCR
    attempts = [
        dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
             use_textline_orientation=False, enable_mkldnn=False),        # 3.x
        dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
             use_textline_orientation=False),                               # 3.x بدون mkldnn
        dict(lang=lang, use_angle_cls=False, show_log=False),               # 2.x
        dict(lang=lang),
    ]
    last = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(str(last))


def _bbox(poly) -> tuple:
    a = np.array(poly, dtype=float).reshape(-1, 2) if np.size(poly) > 4 else None
    if a is None:
        x1, y1, x2, y2 = [float(v) for v in np.array(poly).reshape(-1)[:4]]
        return x1, y1, x2, y2
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _parse(res) -> list:
    """يوحّد مخرجات 2.x/3.x إلى [(x1,y1,x2,y2,text,score)]."""
    out = []
    if res is None:
        return out
    # 3.x: قائمة كائنات شبيهة بالقاموس
    for r in (res if isinstance(res, list) else [res]):
        try:
            d = r if isinstance(r, dict) else getattr(r, "json", None)
            if isinstance(d, dict) and "res" in d and isinstance(d["res"], dict):
                d = d["res"]
            if isinstance(d, dict) and "rec_texts" in d:
                texts = d.get("rec_texts") or []
                scores = d.get("rec_scores") or [1.0] * len(texts)
                polys = d.get("rec_polys")
                if polys is None or len(polys) == 0:
                    polys = d.get("rec_boxes")
                if polys is None or len(polys) == 0:
                    polys = d.get("dt_polys")
                if polys is None:
                    polys = []
                for i, t in enumerate(texts):
                    if i < len(polys):
                        x1, y1, x2, y2 = _bbox(polys[i])
                    else:
                        x1, y1, x2, y2 = 0, i * 10, 100, i * 10 + 8
                    out.append((x1, y1, x2, y2, str(t), float(scores[i])))
                continue
            # 2.x: [[box,(text,score)],...]
            if isinstance(r, list):
                for line in r:
                    if line and len(line) == 2:
                        box, (t, sc) = line[0], line[1]
                        x1, y1, x2, y2 = _bbox(box)
                        out.append((x1, y1, x2, y2, str(t), float(sc)))
        except Exception:  # noqa: BLE001
            continue
    return out


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _compose(items: list) -> str:
    """ترتيب القراءة: أسطر من الأعلى للأسفل؛ داخل السطر يميناً→يساراً إن غلب العربي."""
    if not items:
        return ""
    items = sorted(items, key=lambda t: (t[1] + t[3]) / 2)
    hs = sorted(max(1.0, t[3] - t[1]) for t in items)
    med_h = hs[len(hs) // 2]
    lines, cur, cur_y = [], [], None
    for it in items:
        yc = (it[1] + it[3]) / 2
        if cur and abs(yc - cur_y) > 0.6 * med_h:
            lines.append(cur)
            cur = []
        cur.append(it)
        cur_y = sum((c[1] + c[3]) / 2 for c in cur) / len(cur)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        txt = " ".join(c[4] for c in ln)
        rtl = len(_AR.findall(txt)) > len(_LAT.findall(txt))
        ln = sorted(ln, key=lambda c: -c[0] if rtl else c[0])
        out.append(re.sub(r"\s+", " ", " ".join(c[4] for c in ln)).strip())
    return "\n".join(l for l in out if l)


class SlideOCR:
    def __init__(self, langs: list, min_score: float = 0.5):
        self.langs = langs or ["ar", "en"]
        self.min_score = min_score
        self.engines = {}
        self.ok = True
        for lg in self.langs:
            try:
                self.engines[lg] = _make_paddle(lg)
            except Exception as e:  # noqa: BLE001
                log(f"[ocr] تعذر تهيئة PaddleOCR للغة {lg}: {str(e)[:150]}")
        if not self.engines:
            self.ok = False
            log("[ocr] PaddleOCR غير متاح؛ سيستمر النظام بدون نص الشرائح (تحقق من تثبيت paddlepaddle و paddleocr).")

    def _run(self, eng, img):
        if hasattr(eng, "predict"):
            try:
                return eng.predict(img)
            except Exception:  # noqa: BLE001
                pass
        try:
            return eng.ocr(img, cls=False)
        except TypeError:
            return eng.ocr(img)

    def read(self, image_path) -> str:
        if not self.ok:
            return ""
        img = imread(image_path)
        if img is None:
            return ""
        try:
            img = enhance_for_ocr(img)
        except Exception:  # noqa: BLE001
            pass
        merged: list = []
        for lg, eng in self.engines.items():
            try:
                items = [t for t in _parse(self._run(eng, img)) if t[5] >= self.min_score and t[4].strip()]
            except Exception as e:  # noqa: BLE001
                log(f"[ocr] خطأ أثناء القراءة ({lg}): {str(e)[:100]}")
                continue
            for it in items:
                dup = next((i for i, m in enumerate(merged) if _iou(m, it) > 0.5), None)
                if dup is None:
                    merged.append(it)
                elif it[5] > merged[dup][5]:
                    merged[dup] = it
        return _compose(merged)
