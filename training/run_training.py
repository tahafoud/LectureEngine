"""تدريب كامل بأمر واحد:  تجهيز → (مراجعة) → تقسيم → تدريب → تقييم → تصدير.

    python training/run_training.py --videos data/videos --manual data/manual --out outputs
المخرجات: <out>/models/best.pt  و  <out>/reports/{metrics.json, confusion_matrix_*.png/csv, ...}

--stage prepare  : يتوقف بعد الفرز المبدئي لتراجع المجلدات يدوياً (ثم أعد التشغيل بـ --stage train)
--stage train    : يبدأ من مجلد review الموجود (بعد مراجعتك)
--stage all      : الكل (الافتراضي)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from _common import ROOT

import prepare_data
import split_dataset
import train_cls
from lecture_engine.utils import log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=[str(ROOT / "data" / "videos")])
    ap.add_argument("--manual", default=str(ROOT / "data" / "manual"))
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--min-top1", type=float, default=0.90)
    ap.add_argument("--stage", choices=["all", "prepare", "train"], default="all")
    ap.add_argument("--force", action="store_true", help="إعادة استخراج الإطارات (يمسح مراجعتك اليدوية)")
    a = ap.parse_args(argv)
    work, out = Path(a.work), Path(a.out)
    try:
        if a.stage in ("all", "prepare"):
            log("== 1/3 تجهيز البيانات ==")
            prepare_data.prepare(a.videos, work, a.interval, force=a.force)
            if a.stage == "prepare":
                log(f"تم. راجع {work / 'review'} ثم شغّل: --stage train")
                return 0
        log("== 2/3 تقسيم البيانات ==")
        split_dataset.build_split(work / "review", Path(a.manual), work / "dataset")
        log("== 3/3 التدريب والتقييم ==")
        m = train_cls.run(work / "dataset", out, a.epochs or None, a.min_top1)
        log(f"انتهى. النموذج: {out / 'models' / 'best.pt'} | Top-1(test)={m.get('test', {}).get('top1')}")
        return 0
    except SystemExit as e:
        log(f"⛔ {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        log(f"⛔ خطأ: {str(e)[:300]}")
        if os.environ.get("LE_DEBUG") == "1":
            log(traceback.format_exc()[-1500:])
        return 1


if __name__ == "__main__":
    sys.exit(main())
