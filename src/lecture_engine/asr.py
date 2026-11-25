"""تفريغ الكلام بـ Whisper (faster-whisper) مع تدرّج آمن: GPU fp16 → GPU int8 → CPU."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from .utils import free_gpu, has_cuda, have_ffmpeg, log


def extract_audio(video: Path, out_wav: Path) -> Path:
    """يستخرج المسار الصوتي 16kHz أحادي؛ إن تعذر يُستخدم الفيديو نفسه (faster-whisper يفك ترميزه)."""
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav
    if not have_ffmpeg():
        return video
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)], check=True, timeout=3600,
                       capture_output=True)
        return out_wav if out_wav.exists() and out_wav.stat().st_size > 0 else video
    except Exception as e:  # noqa: BLE001
        log("[asr] تعذر استخراج الصوت بـ ffmpeg (قد لا يحوي الفيديو صوتاً)؛ سيُقرأ الفيديو مباشرة.")
        return video


def _clean(segments: list) -> list:
    out, last, rep = [], "", 0
    for s in segments:
        t = " ".join(s["text"].split())
        if not t:
            continue
        if t == last:
            rep += 1
            if rep >= 2:        # هلوسة تكرار
                continue
        else:
            rep, last = 0, t
        s["text"] = t
        out.append(s)
    return out


class Transcriber:
    def __init__(self, model: str = "auto", language: str = "ar"):
        self.model_name = model
        self.language = language
        self._model = None
        self._key = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _plan(self, size: str) -> list:
        if has_cuda():
            return [(size, "cuda", "float16"), (size, "cuda", "int8_float16"),
                    ("small" if size in ("large-v3", "large-v2", "large", "medium") else size, "cpu", "int8")]
        return [("small" if size in ("large-v3", "large-v2", "large", "medium") else size, "cpu", "int8")]

    def release(self) -> None:
        self._model = None
        self._key = None
        free_gpu()

    def transcribe(self, media: Path, size: Optional[str] = None, duration: float = 0.0,
                   on_progress: Optional[Callable[[float], None]] = None) -> list:
        """يعيد [{"start","end","text"}]. يرفع RuntimeError فقط إذا فشلت كل المحاولات."""
        if not self.available():
            raise RuntimeError("faster-whisper غير مثبت: pip install faster-whisper")
        from faster_whisper import WhisperModel
        size = size or self.model_name
        if size == "auto":
            size = "large-v3" if has_cuda() else "small"
        lang = None if self.language == "auto" else self.language
        last_err = None
        for name, dev, comp in self._plan(size):
            try:
                if self._key != (name, dev, comp):
                    self.release()
                    self._model = WhisperModel(name, device=dev, compute_type=comp)
                    self._key = (name, dev, comp)
                    log(f"[asr] Whisper {name} على {dev} ({comp})")
                segs, info = self._model.transcribe(
                    str(media), language=lang, beam_size=5, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False)
                total = duration or float(getattr(info, "duration", 0) or 0)
                out = []
                for s in segs:
                    if getattr(s, "no_speech_prob", 0) > 0.85 and getattr(s, "avg_logprob", 0) < -1.0:
                        continue
                    if getattr(s, "compression_ratio", 0) > 2.6:
                        continue
                    out.append({"start": round(float(s.start), 2), "end": round(float(s.end), 2),
                                "text": s.text.strip()})
                    if on_progress and total:
                        on_progress(min(1.0, float(s.end) / total))
                return _clean(out)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log(f"[asr] فشلت المحاولة ({name}/{dev}/{comp}): {str(e)[:120]} → محاولة بديلة")
                self.release()
        raise RuntimeError(f"فشل التفريغ الصوتي: {str(last_err)[:200]}")
