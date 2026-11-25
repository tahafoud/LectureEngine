"""اسأل من سطر الأوامر:  python scripts/ask.py "ما هو نظام التشغيل؟" [--lecture L001]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lecture_engine.pipeline import Engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--lecture", default=None)
a = ap.parse_args()
print(json.dumps(Engine().ask(a.question, a.lecture), ensure_ascii=False, indent=2))
