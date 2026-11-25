"""طبقة الدمج: محاذاة زمنية (شريحة ↔ كلام) ثم تصحيح سياقي بـ Qwen، وبناء وحدات المعرفة."""
from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from .utils import fmt_time

SYSTEM_FIX = (
    "أنت مصحّح لغوي لتفريغ صوتي لمحاضرة جامعية. ستُعطى نص الشريحة (مرجع للمصطلحات العلمية والتقنية) "
    "ونص التفريغ الصوتي لنفس الفترة. صحّح أخطاء التفريغ فقط (المصطلحات، الأسماء، الأرقام، الإملاء) "
    "اعتماداً على نص الشريحة. لا تضف معلومات جديدة ولا تحذف شيئاً ولا تلخّص ولا تترجم، "
    "وأبقِ الكلمات الإنجليزية كما هي. أعد النص المصحح فقط دون أي مقدمة أو شرح."
)


def assign_segments(slides: list, segments: list) -> dict:
    """يسند كل مقطع كلام إلى الشريحة التي تغطي منتصفه الزمني (قبل الأولى → الأولى، بعد الأخيرة → الأخيرة)."""
    out = {s["slide_id"]: [] for s in slides}
    if not slides:
        return out
    starts = [s["start_time"] for s in slides]
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        k = 0
        for i, st in enumerate(starts):
            if mid >= st:
                k = i
        out[slides[k]["slide_id"]].append(seg)
    return out


def chunk_segments(segs: list, max_chars: int = 450) -> list:
    chunks, cur = [], None
    for s in segs:
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
            continue
        if len(cur["text"]) + len(s["text"]) + 1 > max_chars and len(cur["text"]) >= 120:
            chunks.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
        else:
            cur["text"] += " " + s["text"]
            cur["end"] = s["end"]
    if cur:
        chunks.append(cur)
    return chunks


def _guard(raw: str, out: Optional[str]) -> str:
    """يقبل ناتج Qwen فقط إن بقي قريباً من الأصل (منع الهلوسة والتلخيص)."""
    if not out:
        return raw
    out = re.sub(r"^(النص المصحح|النص المصحَّح|التصحيح)\s*[:：]\s*", "", out.strip())
    out = out.strip('"“”«» \n')
    if not out:
        return raw
    ratio = len(out) / max(1, len(raw))
    if ratio < 0.6 or ratio > 1.5:
        return raw
    if difflib.SequenceMatcher(None, raw, out).ratio() < 0.55:
        return raw
    return out


def correct_text(llm, slide_text: str, raw: str) -> str:
    if llm is None or not getattr(llm, "available", False) or len(raw) < 15:
        return raw
    user = (f"نص الشريحة:\n{(slide_text or '(لا يوجد نص)')[:800]}\n\n"
            f"التفريغ الصوتي:\n{raw}\n\nالنص المصحح:")
    out = llm.chat(SYSTEM_FIX, user, max_new_tokens=min(1024, int(len(raw) * 1.6) + 64))
    return _guard(raw, out)


def build_units(lecture_id: str, title: str, slides: list, segments: list, llm=None,
                on_progress: Optional[Callable[[float], None]] = None) -> list:
    """وحدات المعرفة: (lecture_id, slide_id, timestamp, slide_image_path, ocr_text, corrected_audio_text)."""
    units = []
    if not slides:      # فيديو بلا شرائح: وحدات صوتية فقط
        slides = [{"slide_id": 0, "start_time": 0.0, "end_time": 0.0, "image": "", "ocr_text": ""}]
    assigned = assign_segments(slides, segments)
    total = max(1, sum(len(chunk_segments(assigned[s["slide_id"]])) for s in slides))
    done = 0
    for sl in slides:
        sid = sl["slide_id"]
        chunks = chunk_segments(assigned[sid])
        if not chunks:
            units.append({"unit_id": f"{lecture_id}_s{sid:03d}_00", "lecture_id": lecture_id,
                          "lecture_title": title, "slide_id": sid, "timestamp": sl["start_time"],
                          "end_timestamp": sl["end_time"], "slide_image_path": sl["image"],
                          "ocr_text": sl.get("ocr_text", ""), "raw_audio_text": "",
                          "corrected_audio_text": ""})
            continue
        for k, ch in enumerate(chunks):
            corrected = correct_text(llm, sl.get("ocr_text", ""), ch["text"])
            units.append({"unit_id": f"{lecture_id}_s{sid:03d}_{k:02d}", "lecture_id": lecture_id,
                          "lecture_title": title, "slide_id": sid, "timestamp": ch["start"],
                          "end_timestamp": ch["end"], "slide_image_path": sl["image"],
                          "ocr_text": sl.get("ocr_text", ""), "raw_audio_text": ch["text"],
                          "corrected_audio_text": corrected})
            done += 1
            if on_progress:
                on_progress(done / total)
    return units


def unit_label(u: dict) -> str:
    return f"المحاضرة {u['lecture_id']}، الشريحة {u['slide_id']}، الدقيقة {fmt_time(u['timestamp'])}"
