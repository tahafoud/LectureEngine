from __future__ import annotations

import gc
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def fmt_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def is_url(s: str) -> bool:
    return str(s).strip().lower().startswith(("http://", "https://"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def imread(path) -> "np.ndarray | None":
    """قراءة صورة تدعم المسارات العربية/يونيكود."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def imwrite(path, img, quality: int = 92) -> bool:
    try:
        ext = Path(path).suffix.lower() or ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except Exception:  # noqa: BLE001
        return False


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


_AR_DIAC = re.compile(r"[ً-ْـ]")


def normalize_ar(text: str) -> str:
    t = _AR_DIAC.sub("", text or "")
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return t.lower()


def tokens(text: str) -> list:
    return [w for w in re.findall(r"[\w]+", normalize_ar(text)) if len(w) > 1]
