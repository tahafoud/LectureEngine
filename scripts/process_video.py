"""معالجة فيديو/رابط من سطر الأوامر دون تشغيل الخادم.

    python scripts/process_video.py data/videos/lec1.mp4 --title "نظم التشغيل - 1"
    python scripts/process_video.py "https://youtu.be/XXXX"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lecture_engine.pipeline import Engine  # noqa: E402
from lecture_engine.utils import is_url  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="+", help="مسار فيديو أو رابط (يمكن أكثر من واحد)")
    ap.add_argument("--title", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    eng = Engine()
    rc = 0
    for src in a.source:
        title = a.title or (src if is_url(src) else Path(src).stem)
        lid = eng.submit(title, src, "url" if is_url(src) else "file")
        eng.process(lid, force=a.force)
        j = eng.jobs.get(lid)
        print(f"{lid}: {j['status']} | شرائح={j['slides']} وحدات={j['units']} | {j['error'] or ''}")
        for w in j["warnings"]:
            print("  تنبيه:", w)
        rc |= j["status"] != "done"
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
