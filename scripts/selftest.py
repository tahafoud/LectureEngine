"""فحص جاهزية البيئة (لا يرفع أخطاء أبداً؛ يطبع ✅/⚠️ فقط).  python scripts/selftest.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check(name, fn):
    try:
        msg = fn()
        print(f"✅ {name}" + (f": {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ {name}: {str(e)[:160]}")


def torch_cuda():
    import torch
    return f"torch {torch.__version__} | CUDA={torch.cuda.is_available()}"


def ultra():
    import ultralytics
    return ultralytics.__version__


def whisper():
    from faster_whisper import WhisperModel
    import torch
    if torch.cuda.is_available():
        try:
            WhisperModel("tiny", device="cuda", compute_type="float16")
            return "يعمل على GPU"
        except Exception as e:  # noqa: BLE001
            WhisperModel("tiny", device="cpu", compute_type="int8")
            return f"GPU غير صالح ({str(e)[:60]}) → سيعمل على CPU تلقائياً"
    WhisperModel("tiny", device="cpu", compute_type="int8")
    return "يعمل على CPU"


def paddle():
    import cv2
    import numpy as np
    from lecture_engine.ocr import SlideOCR
    img = np.full((200, 800, 3), 255, np.uint8)
    cv2.putText(img, "Operating System", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        cv2.imwrite(str(p), img)
        o = SlideOCR(["en"])
        if not o.ok:
            raise RuntimeError("تعذر تهيئة PaddleOCR")
        return f"قرأ: {o.read(p)!r}"


def chroma():
    import chromadb
    return chromadb.__version__


def st():
    import sentence_transformers
    return sentence_transformers.__version__


def tf():
    import transformers
    return transformers.__version__


for n, f in [("PyTorch", torch_cuda), ("Ultralytics", ultra), ("faster-whisper", whisper),
             ("PaddleOCR", paddle), ("ChromaDB", chroma), ("sentence-transformers", st),
             ("transformers (Qwen)", tf)]:
    check(n, f)
print("انتهى الفحص. العناصر ⚠️ لا توقف النظام: يعمل بدائل آمنة، لكن الأفضل إصلاحها.")
