"""تقسيم train 80% / val 10% / test 10% بلا تسرّب + صور يدوية + دعم الفئات الأقل تمثيلاً.

منع التسرب:
  * إن وُجد ≥3 فيديوهات: التقسيم على مستوى الفيديو (إطارات الفيديو الواحد كلها في قسم واحد).
  * وإلا: كتل زمنية متتالية لكل فئة مع حذف إطار فاصل بين الكتل.
  * الصور اليدوية (data/manual/<class>/) تدخل التدريب فقط (أصلها غير معروف فلا نخاطر بالتسرب).
  * الصور الاصطناعية (blank/transition) للتدريب فقط، ولا تدخل val/test أبداً.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from _common import CLASSES, IMG_EXT, ROOT  # noqa: F401

from lecture_engine.utils import imread, imwrite, log, write_json


def _collect(review: Path) -> list:
    items = []
    for c in CLASSES:
        for p in sorted((review / c).glob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            vid, _, rest = p.stem.partition("__")
            try:
                t = int(rest) / 1000.0
            except ValueError:
                t = 0.0
            items.append({"cls": c, "video": vid if rest else "x_" + p.stem, "t": t, "path": p})
    return items


def _video_split(items: list, seed_tries: int = 400):
    vids = sorted({i["video"] for i in items})
    if len(vids) < 3:
        return None
    n_val = n_test = max(1, round(0.1 * len(vids)))
    present = {c for c in CLASSES if any(i["cls"] == c for i in items)}
    counts = {c: {v: 0 for v in vids} for c in present}
    for i in items:
        counts[i["cls"]][i["video"]] += 1
    best, best_score = None, None
    for s in range(seed_tries):
        rnd = random.Random(s)
        order = vids[:]
        rnd.shuffle(order)
        val, test, train = set(order[:n_val]), set(order[n_val:n_val + n_test]), set(order[n_val + n_test:])
        # تغطية: كل فئة موجودة تظهر في val وtest (≥2 صورة) والتدريب
        cov = 0
        err = 0.0
        for c in present:
            tot = sum(counts[c].values())
            if tot < 6:
                continue
            for part in (val, test):
                if sum(counts[c][v] for v in part) < 2:
                    cov += 1
            if sum(counts[c][v] for v in train) < 0.5 * tot:
                cov += 1
            for part, share in ((val, 0.1), (test, 0.1)):
                err += abs(sum(counts[c][v] for v in part) / tot - share)
        score = (cov, err)
        if best_score is None or score < best_score:
            best, best_score = (train, val, test), score
        if cov == 0 and err < 0.05 * len(present):
            break
    if best_score[0] > 0:
        return None         # تعذر ضمان التغطية على مستوى الفيديو
    train, val, test = best
    return {"train": [i for i in items if i["video"] in train],
            "val": [i for i in items if i["video"] in val],
            "test": [i for i in items if i["video"] in test]}


def _block_split(items: list) -> dict:
    out = {"train": [], "val": [], "test": []}
    for c in CLASSES:
        grp = sorted([i for i in items if i["cls"] == c], key=lambda i: (i["video"], i["t"]))
        n = len(grp)
        if n == 0:
            continue
        if n < 5:
            out["train"] += grp
            continue
        n_val = n_test = max(1, int(round(0.1 * n)))
        n_train = n - n_val - n_test
        out["train"] += grp[:max(1, n_train - 1)]            # إطار فاصل محذوف
        out["val"] += grp[n_train:n_train + n_val]
        out["test"] += grp[n_train + n_val:]
    return out


def _synth_blank(rnd: random.Random, k: int) -> list:
    imgs = []
    for _ in range(k):
        base = rnd.choice([rnd.randint(0, 20), rnd.randint(236, 255)])
        img = np.full((360, 640, 3), base, np.uint8)
        noise = np.random.default_rng(rnd.randint(0, 10**6)).normal(0, rnd.uniform(0, 2.5), img.shape)
        imgs.append(np.clip(img + noise, 0, 255).astype(np.uint8))
    return imgs


def _synth_transition(rnd: random.Random, slides: list, k: int) -> list:
    imgs = []
    for _ in range(k):
        a = imread(rnd.choice(slides)["path"])
        b = imread(rnd.choice(slides)["path"])
        if a is None or b is None:
            continue
        a, b = cv2.resize(a, (640, 360)), cv2.resize(b, (640, 360))
        al = rnd.uniform(0.3, 0.7)
        m = cv2.addWeighted(a, al, b, 1 - al, 0)
        if rnd.random() < 0.6:
            kk = rnd.choice([9, 15, 21])
            m = cv2.GaussianBlur(m, (kk, kk), 0)
        imgs.append(m)
    return imgs


def build_split(review: Path, manual: Path, out: Path, seed: int = 0, synth_target: int = 40) -> dict:
    items = _collect(review)
    totals = {c: sum(1 for i in items if i["cls"] == c) for c in CLASSES}
    if totals["slide"] < 10:
        raise SystemExit(f"عدد صور slide قليل جداً ({totals['slide']}). أضف فيديوهات أكثر أو راجع مجلد review.")

    sp = _video_split(items)
    strategy = "video-level"
    if sp is None:
        sp, strategy = _block_split(items), "time-blocks (تقييم أضعف: فيديوهات قليلة)"
    warnings = []

    shutil.rmtree(out, ignore_errors=True)
    for part in ("train", "val", "test"):
        for c in CLASSES:
            (out / part / c).mkdir(parents=True, exist_ok=True)

    def put(part, it):
        shutil.copy2(it["path"], out / part / it["cls"] / it["path"].name)

    for part in ("train", "val", "test"):
        for it in sp[part]:
            put(part, it)

    # صور يدوية → تدريب فقط
    n_manual = 0
    for c in CLASSES:
        for p in sorted((manual / c).glob("*")) if (manual / c).exists() else []:
            if p.suffix.lower() in IMG_EXT:
                shutil.copy2(p, out / "train" / c / f"manual__{p.name}")
                n_manual += 1

    # صور اصطناعية للفئات الشحيحة (تدريب فقط)
    rnd = random.Random(seed)
    synth = {}
    n_train = {c: len(list((out / "train" / c).glob("*"))) for c in CLASSES}
    train_slides = [{"path": p} for p in (out / "train" / "slide").glob("*")]
    for c in ("blank", "transition"):
        need = max(0, synth_target - n_train[c])
        if need == 0:
            continue
        imgs = _synth_blank(rnd, need) if c == "blank" else _synth_transition(rnd, train_slides, need)
        for k, im in enumerate(imgs):
            imwrite(out / "train" / c / f"synth__{k:03d}.jpg", im, 90)
        synth[c] = len(imgs)
        warnings.append(f"فئة {c}: {n_train[c]} صورة حقيقية فقط في التدريب؛ أُضيفت {len(imgs)} صورة اصطناعية.")

    counts = {part: {c: len(list((out / part / c).glob("*"))) for c in CLASSES} for part in ("train", "val", "test")}
    for part in ("val", "test"):
        for c in CLASSES:
            if counts[part][c] == 0:
                warnings.append(f"لا صور حقيقية لفئة {c} في {part}؛ ستكون مقاييسها غير ممثِّلة لهذه الفئة.")
    report = {"strategy": strategy, "counts": counts, "manual_added": n_manual, "synthetic_added": synth,
              "warnings": warnings}
    write_json(out.parent / "dataset_report.json", report)
    log(f"[split] {strategy} | {counts}")
    for w in warnings:
        log("[split] ⚠️ " + w)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--manual", default=str(ROOT / "data" / "manual"))
    a = ap.parse_args(argv)
    w = Path(a.work)
    build_split(w / "review", Path(a.manual), w / "dataset")


if __name__ == "__main__":
    main()
