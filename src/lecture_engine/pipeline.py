"""المحرّك: يربط كل المراحل من الرابط/الملف حتى الفهرسة والاستعلام."""
from __future__ import annotations

import os
import shutil
import threading
import traceback
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings
from .fusion import build_units
from .ingest import fetch_video, find_video
from .jobs import JobStore
from .utils import fmt_time, log, read_json, write_json


class Engine:
    """تُحمَّل النماذج كسولاً (عند أول استخدام) لتوفير الذاكرة."""

    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self.s.lectures_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = JobStore(self.s.db_path)
        self._c: dict = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- مكوّنات كسولة
    def _get(self, key, factory):
        with self._lock:
            if key not in self._c:
                self._c[key] = factory()
            return self._c[key]

    @property
    def classifier(self):
        from .vision.classifier import SlideClassifier
        return self._get("cls", lambda: SlideClassifier(self.s.yolo_weights))

    @property
    def ocr(self):
        from .ocr import SlideOCR
        return self._get("ocr", lambda: SlideOCR(self.s.ocr_langs, self.s.ocr_min_score))

    @property
    def asr(self):
        from .asr import Transcriber
        return self._get("asr", lambda: Transcriber(self.s.asr_model, self.s.asr_language))

    @property
    def llm(self):
        from .llm import QwenLLM
        return self._get("llm", lambda: QwenLLM(self.s))

    @property
    def index(self):
        from .index import LectureIndex
        return self._get("index", lambda: LectureIndex(self.s))

    # ----------------------------------------------------------- واجهة عامة
    def submit(self, title: str, source: str, kind: str = "file") -> str:
        lid = self.jobs.create(title or "محاضرة", source, kind)
        (self.s.lectures_dir / lid).mkdir(parents=True, exist_ok=True)
        return lid

    def _set(self, lid: str, stage: str, progress: float, message: str = "", **kw) -> None:
        self.jobs.update(lid, stage=stage, progress=round(progress, 1), message=message, **kw)

    def process(self, lid: str, force: bool = False) -> None:
        job = self.jobs.get(lid)
        if not job:
            return
        ldir = self.s.lectures_dir / lid
        warnings: list = []
        try:
            self.jobs.update(lid, status="processing", error="")
            # 1) جلب الفيديو
            self._set(lid, "download", 2, "جلب الفيديو")
            video = find_video(ldir) or fetch_video(job["source"], ldir)

            # 2) الرؤية
            slides = self._vision(lid, ldir, video, force)
            duration = max([s["end_time"] for s in slides] or [0.0])
            self.jobs.update(lid, slides=len(slides), duration=duration)
            if not slides:
                warnings.append("لم يُكتشف أي شريحة؛ سيُفهرس الصوت فقط.")

            # 3) OCR
            self._ocr(lid, ldir, slides, warnings)

            # 4) الصوت
            segments = self._audio(lid, ldir, video, duration, warnings, force)

            # 5) الدمج والتصحيح السياقي
            self._set(lid, "fusion", 80, "تصحيح التفريغ بمساعدة نص الشريحة (Qwen)")
            title = job["title"]
            units = build_units(lid, title, slides, segments, self.llm,
                                on_progress=lambda f: self._set(lid, "fusion", 80 + 12 * f))
            write_json(ldir / "units.json", units)

            # 6) الفهرسة
            self._set(lid, "index", 93, "الفهرسة في ChromaDB")
            self.index.delete_lecture(lid)
            n = self.index.add_units(units)
            write_json(ldir / "meta.json", {"lecture_id": lid, "title": title, "slides": len(slides),
                                            "units": n, "duration": duration, "warnings": warnings})
            self._set(lid, "done", 100, "اكتملت المعالجة", status="done", units=n, warnings=warnings)
            log(f"[pipeline] {lid} ✓ شرائح={len(slides)} وحدات={n}")
        except Exception as e:  # noqa: BLE001
            log(f"[pipeline] {lid} ✗ {e}")
            if os.environ.get("LE_DEBUG") == "1":
                log(traceback.format_exc()[-1500:])
            self.jobs.update(lid, status="failed", stage="failed", error=str(e)[:500], warnings=warnings)
        finally:
            try:
                self.asr.release()
            except Exception:  # noqa: BLE001
                pass

    def _vision(self, lid: str, ldir: Path, video: Path, force: bool) -> list:
        sf = ldir / "slides.json"
        cached = read_json(sf)
        if cached is not None and not force and all((ldir / s["image"]).exists() for s in cached):
            return cached
        from .vision.frames import extract_frames, video_duration
        from .vision.segmenter import build_slides, save_slides
        from .vision.similarity import prelabel
        iv = self.s.frame_interval
        fdir = ldir / "_frames"
        shutil.rmtree(fdir, ignore_errors=True)
        self._set(lid, "frames", 5, "استخراج الإطارات")
        items = extract_frames(video, fdir, interval=iv,
                               on_progress=lambda f: self._set(lid, "frames", 5 + 15 * f))
        dur = video_duration(video) or (items[-1][0] + iv)
        self._set(lid, "classify", 20, "تصنيف الإطارات (slide/blank/transition)")
        paths = [p for _, p in items]
        labels = self.classifier.classify(paths)
        samples = [{"t": t, "path": p, "label": lb} for (t, p), lb in zip(items, labels)]
        segs = build_slides(samples, iv, dur, self.s.slide_change_threshold, self.s.min_slide_frames)
        if not segs:    # احتياط: النموذج لم يجد شرائح → جرّب القواعد الخفيفة
            labels2 = prelabel(paths)
            segs = build_slides([{"t": t, "path": p, "label": lb} for (t, p), lb in zip(items, labels2)],
                                iv, dur, self.s.slide_change_threshold, self.s.min_slide_frames)
        slides = save_slides(segs, ldir / "slides")
        shutil.rmtree(fdir, ignore_errors=True)
        write_json(sf, slides)
        return slides

    def _ocr(self, lid: str, ldir: Path, slides: list, warnings: list) -> None:
        todo = [s for s in slides if not s.get("ocr_done")]
        if not todo:
            return
        ocr = self.ocr
        if not ocr.ok:
            warnings.append("PaddleOCR غير متاح؛ لم يُستخرج نص الشرائح.")
            return
        for i, s in enumerate(todo, 1):
            self._set(lid, "ocr", 40 + 15 * i / len(todo), f"قراءة نص الشريحة {s['slide_id']}")
            try:
                s["ocr_text"] = ocr.read(ldir / s["image"])
            except Exception as e:  # noqa: BLE001
                s["ocr_text"] = ""
                log(f"[ocr] شريحة {s['slide_id']}: {str(e)[:80]}")
            s["ocr_done"] = True
        write_json(ldir / "slides.json", slides)

    def _audio(self, lid: str, ldir: Path, video: Path, duration: float, warnings: list, force: bool) -> list:
        tf = ldir / "transcript.json"
        cached = read_json(tf)
        if cached is not None and not force:
            return cached.get("segments", [])
        from .asr import extract_audio
        self._set(lid, "asr", 55, "استخراج الصوت وتفريغه (Whisper)")
        try:
            media = extract_audio(video, ldir / "audio.wav")
            segs = self.asr.transcribe(media, duration=duration,
                                       on_progress=lambda f: self._set(lid, "asr", 55 + 25 * f))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"تعذر التفريغ الصوتي: {str(e)[:150]}")
            return []
        write_json(tf, {"language": self.s.asr_language, "segments": segs})
        wav = ldir / "audio.wav"
        if wav.exists():
            wav.unlink()
        # srt للمراجعة
        try:
            def ts(x):
                h, r = divmod(int(x), 3600); m, s = divmod(r, 60)
                return f"{h:02d}:{m:02d}:{s:02d},{int((x % 1) * 1000):03d}"
            (ldir / "transcript.srt").write_text(
                "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(segs, 1)),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return segs

    # ----------------------------------------------------------- استعلام
    def ask(self, question: str, lecture_id: Optional[str] = None, top_k: Optional[int] = None) -> dict:
        from .qa import answer
        return answer(self.index, self.llm, question, lecture_id, top_k or self.s.top_k)

    def transcribe_question(self, audio_path: Path) -> str:
        segs = self.asr.transcribe(Path(audio_path), size=self.s.asr_query_model)
        return " ".join(s["text"] for s in segs).strip()

    def delete(self, lid: str) -> None:
        try:
            self.index.delete_lecture(lid)
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.s.lectures_dir / lid, ignore_errors=True)
        self.jobs.delete(lid)
