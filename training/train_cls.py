"""ضبط دقيق لـ YOLOv8n-cls + تقييم (Top-1/Top-5/Confusion Matrix) على val وtest."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np

from _common import CLASSES, IMG_EXT, ROOT  # noqa: F401

from lecture_engine.utils import has_cuda, log, write_json

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("YOLO_VERBOSE", "False")


def train(dataset: Path, run_dir: Path, epochs=None, imgsz: int = 224, base: str = "yolov8n-cls.pt") -> Path:
    from ultralytics import YOLO
    gpu = has_cuda()
    epochs = epochs or (30 if gpu else 15)
    batch = 64 if gpu else 16
    last = None
    for attempt in range(3):
        try:
            model = YOLO(base)
            model.train(data=str(dataset), epochs=epochs, imgsz=imgsz, batch=batch, device=0 if gpu else "cpu",
                        project=str(run_dir), name="cls", exist_ok=True, workers=2, patience=10,
                        plots=True, verbose=False, amp=(attempt == 0 and gpu), seed=0)
            best = run_dir / "cls" / "weights" / "best.pt"
            if best.exists():
                return best
            raise RuntimeError("لم يُنتج التدريب best.pt")
        except Exception as e:  # noqa: BLE001 - نفاد الذاكرة/AMP: نخفّف ونعيد
            last = e
            batch = max(8, batch // 2)
            log(f"[train] تعثّر ({str(e)[:120]}) → إعادة بدفعة {batch}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    raise SystemExit(f"فشل التدريب بعد 3 محاولات: {last}")


def evaluate_split(model, split_dir: Path, names: dict) -> dict:
    files, truth = [], []
    for ci, c in names.items():
        for p in sorted((split_dir / c).glob("*")) if (split_dir / c).exists() else []:
            if p.suffix.lower() in IMG_EXT:
                files.append(str(p))
                truth.append(int(ci))
    if not files:
        return {"n": 0}
    nc = len(names)
    cm = np.zeros((nc, nc), dtype=int)
    top1 = top5 = 0
    for i in range(0, len(files), 64):
        for r, t in zip(model.predict(files[i:i + 64], imgsz=224, verbose=False), truth[i:i + 64]):
            pred = int(r.probs.top1)
            cm[t, pred] += 1
            top1 += pred == t
            top5 += t in [int(x) for x in r.probs.top5[:5]]
    n = len(files)
    labels = [names[i] for i in range(nc)]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {"n": n, "top1": round(top1 / n, 4), "top5": round(top5 / n, 4), "labels": labels,
            "confusion_matrix": cm.tolist(), "confusion_matrix_pct": (norm * 100).round(1).tolist()}


def save_cm(res: dict, out_png: Path, out_csv: Path, title: str) -> None:
    cm = np.array(res["confusion_matrix"])
    pct = np.array(res["confusion_matrix_pct"])
    labels = res["labels"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, l in enumerate(labels):
            w.writerow([l] + [f"{cm[i, j]} ({pct[i, j]}%)" for j in range(len(labels))])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nTop-1={res['top1']:.3f}  Top-5={res['top5']:.3f}  (n={res['n']})", fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center",
                        color="white" if pct[i, j] > 50 else "black", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"[eval] تعذر رسم المصفوفة: {e}")


def run(dataset: Path, out: Path, epochs=None, min_top1: float = 0.90) -> dict:
    from ultralytics import YOLO
    run_dir = dataset.parent / "runs"
    best = train(dataset, run_dir, epochs)
    model = YOLO(str(best))
    names = {int(k): v for k, v in dict(model.names).items()}
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out / "models" / "best.pt")
    metrics = {"classes": list(names.values())}
    for split in ("val", "test"):
        res = evaluate_split(model, dataset / split, names)
        metrics[split] = res
        if res.get("n"):
            save_cm(res, out / "reports" / f"confusion_matrix_{split}.png",
                    out / "reports" / f"confusion_matrix_{split}.csv", f"YOLOv8n-cls — {split}")
            log(f"[eval] {split}: Top-1={res['top1']:.3f} Top-5={res['top5']:.3f} (n={res['n']})")
    ref = metrics.get("test", {}).get("top1", metrics.get("val", {}).get("top1", 0.0))
    metrics["min_top1"] = min_top1
    metrics["passed"] = bool(ref >= min_top1)
    if not metrics["passed"]:
        log(f"[eval] ⚠️ Top-1={ref:.3f} أقل من الهدف {min_top1}: النموذج محفوظ لكن راجع الصور في review وأضف بيانات.")
    write_json(out / "reports" / "metrics.json", metrics)
    for f in (run_dir / "cls").glob("*.png"):          # منحنيات التدريب
        shutil.copy2(f, out / "reports" / f.name)
    if (run_dir / "cls" / "results.csv").exists():
        shutil.copy2(run_dir / "cls" / "results.csv", out / "reports" / "training_results.csv")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--min-top1", type=float, default=0.90)
    a = ap.parse_args(argv)
    run(Path(a.work) / "dataset", Path(a.out), a.epochs or None, a.min_top1)


if __name__ == "__main__":
    main()
