"""مصنّف الإطارات: YOLOv8n-cls المدرَّب (best.pt) مع بديل بقواعد خفيفة إن لم يتوفر النموذج."""
from __future__ import annotations

from pathlib import Path

from ..utils import imread, log
from .similarity import _feat, is_blank_stats, prelabel

CLASSES = ["slide", "blank", "transition"]


class SlideClassifier:
    def __init__(self, weights=None):
        self.model = None
        self.slide_idx = None
        if weights and Path(weights).exists():
            try:
                from ultralytics import YOLO
                m = YOLO(str(weights))
                names = {int(k): v for k, v in dict(m.names).items()}
                if "slide" in names.values():
                    self.model = m
                    self.slide_idx = [k for k, v in names.items() if v == "slide"][0]
                    self.names = names
                    log(f"[vision] نموذج التصنيف: {weights} | الفئات: {list(names.values())}")
                else:
                    log("[vision] النموذج لا يحوي فئة slide؛ سيُستخدم الكشف بالقواعد.")
            except Exception as e:  # noqa: BLE001
                log(f"[vision] تعذر تحميل النموذج ({str(e)[:120]})؛ سيُستخدم الكشف بالقواعد.")
        else:
            log("[vision] لا يوجد best.pt؛ سيُستخدم الكشف بالقواعد (ضع النموذج المدرَّب في models/best.pt).")

    @property
    def available(self) -> bool:
        return self.model is not None

    def classify(self, paths: list, batch: int = 64) -> list:
        """يعيد قائمة تسميات بنفس ترتيب paths."""
        if not paths:
            return []
        if self.model is None:
            return prelabel(paths)
        labels = []
        i = 0
        b = batch
        while i < len(paths):
            chunk = [str(p) for p in paths[i:i + b]]
            try:
                res = self.model.predict(chunk, imgsz=224, verbose=False)
                for r in res:
                    labels.append(self.names[int(r.probs.top1)])
                i += len(chunk)
            except Exception as e:  # noqa: BLE001 - نفاد الذاكرة وغيره
                if b > 4:
                    b //= 2
                    continue
                log(f"[vision] فشل التصنيف ({str(e)[:100]})؛ استخدام القواعد لباقي الإطارات.")
                labels.extend(prelabel(paths[i:]))
                break
        # الشاشة الفارغة تُكتشف بالقواعد بدقة عالية: نصحح بها أخطاء النموذج
        for j, p in enumerate(paths[:len(labels)]):
            if labels[j] != "blank":
                f = _feat(p)
                if f and is_blank_stats(f["mean"], f["std"]):
                    labels[j] = "blank"
        return labels
