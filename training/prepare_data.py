"""مرحلة تجهيز البيانات (للتدريب فقط، لا تُستخدم في التشغيل النهائي):

فيديوهات/روابط → إطار كل ثانيتين → حذف المتكرر (SSIM) → فرز مبدئي بقواعد خفيفة إلى:
    <work>/review/slide/   <work>/review/blank/   <work>/review/transition/
راجع المجلدات بسرعة (انقل أي صورة خاطئة إلى مجلدها الصحيح أو احذفها) ثم شغّل split/train.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _common import CLASSES, ROOT  # noqa: F401

from lecture_engine.ingest import VIDEO_EXT, fetch_video
from lecture_engine.utils import imread, imwrite, is_url, log, write_json
from lecture_engine.vision.frames import extract_frames
from lecture_engine.vision.similarity import diff_frac, prelabel, thumb_of

CAPS = {"slide": 400, "blank": 80, "transition": 120}      # أقصى عدد لكل فيديو لكل فئة


def discover(inputs: list, dl_dir: Path) -> list:
    vids = []
    for x in inputs:
        if is_url(x):
            d = dl_dir / f"url_{abs(hash(x)) % 10**8}"
            try:
                vids.append(fetch_video(x, d))
            except Exception as e:  # noqa: BLE001
                log(f"[prep] تخطي الرابط ({str(e)[:100]})")
            continue
        p = Path(x)
        if p.is_dir():
            vids += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT)
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    vids += discover([line], dl_dir)
        elif p.is_file():
            vids.append(p)
    return list(dict.fromkeys(vids))


def _subsample(items: list, cap: int) -> list:
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def prepare(inputs: list, work: Path, interval: float = 2.0, dedupe_thr: float = 0.002, force: bool = False) -> dict:
    review = work / "review"
    if review.exists() and any(review.rglob("*.jpg")) and not force:
        log(f"[prep] مجلد المراجعة موجود ({review}) وسيُحتفظ بما فيه (مراجعتك اليدوية محفوظة). استخدم --force لإعادته.")
        return {"reused": True}
    shutil.rmtree(review, ignore_errors=True)
    for c in CLASSES:
        (review / c).mkdir(parents=True, exist_ok=True)
    videos = discover(inputs, work / "_downloads")
    if not videos:
        raise SystemExit("لا توجد فيديوهات. ضعها في مجلد videos أو مرّر مساراً صحيحاً.")
    log(f"[prep] {len(videos)} فيديو")
    manifest, stats = [], {}
    for vi, v in enumerate(videos, 1):
        vid = f"v{vi:02d}"
        log(f"[prep] ({vi}/{len(videos)}) {Path(v).name}")
        try:
            tmp = work / "_frames" / vid
            items = extract_frames(v, tmp, interval=interval, max_side=960)
        except Exception as e:  # noqa: BLE001
            log(f"[prep] تخطي {Path(v).name}: {str(e)[:100]}")
            continue
        labels = prelabel([p for _, p in items])
        kept = {c: [] for c in CLASSES}
        last_thumb = {c: None for c in CLASSES}
        for (t, p), lb in zip(items, labels):
            img = imread(p)
            if img is None:
                continue
            th = thumb_of(img)
            if last_thumb[lb] is not None and diff_frac(last_thumb[lb], th) <= dedupe_thr:
                continue                       # مكرر (تغيّر ≤ 0.2% من المساحة)
            last_thumb[lb] = th
            kept[lb].append((t, img))
        stats[vid] = {"name": Path(v).name}
        for c in CLASSES:
            for t, img in _subsample(kept[c], CAPS[c]):
                name = f"{vid}__{int(round(t * 1000)):08d}.jpg"
                imwrite(review / c / name, img, 90)
                manifest.append({"file": name, "class": c, "video": vid, "t": t})
            stats[vid][c] = min(len(kept[c]), CAPS[c])
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(work / "_frames", ignore_errors=True)
    write_json(work / "prep_manifest.json", {"videos": stats, "items": manifest})
    tot = {c: sum(s.get(c, 0) for s in stats.values()) for c in CLASSES}
    log(f"[prep] الناتج: {tot}  → راجع {review}")
    return {"videos": stats, "totals": tot}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    prepare(a.videos, Path(a.work), a.interval, force=a.force)


if __name__ == "__main__":
    main()
