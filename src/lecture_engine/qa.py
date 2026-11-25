"""سؤال وجواب (RAG): بحث دلالي + إعادة ترتيب معجمية + توليد بـ Qwen أو استخراج جمل مع المصدر."""
from __future__ import annotations

import re
from typing import Optional

from .fusion import unit_label
from .utils import fmt_time, tokens

SYSTEM_QA = (
    "أنت مساعد جامعي لطلاب جامعة ذمار. أجب عن سؤال الطالب بالعربية بشكل واضح ومباشر اعتماداً فقط على "
    "المقاطع المرفقة من المحاضرة (نص الشريحة وشرح المحاضر). لا تستخدم معلومات من خارجها. "
    "إن لم تجد الإجابة في المقاطع فقل صراحة: لم أجد إجابة هذا السؤال في المحاضرات المفهرسة. "
    "اذكر رقم المقطع الذي استندت إليه بين أقواس مربعة مثل [1]."
)
_STOP = set("ما ماذا هو هي هل في من على الى الي عن مع هذا هذه ذلك التي الذي كيف لماذا متى اين كم او ان ثم قد كان يكون the a an of is are what how why".split())


def _qtokens(q: str) -> set:
    return {t for t in tokens(q) if t not in _STOP}


def rerank(question: str, hits: list, top_k: int) -> list:
    qt = _qtokens(question)
    for h in hits:
        text = " ".join(tokens(h.get("ocr_text", "") + " " + (h.get("corrected_audio_text") or "")))
        words = set(text.split())
        ov = len(qt & words) / len(qt) if qt else 0.0
        h["lexical"] = ov
        h["final"] = h["score"] + 0.25 * ov
    return sorted(hits, key=lambda h: -h["final"])[:top_k]


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?؟۔\n])\s+", text) if len(s.strip()) > 8]


def extractive_answer(question: str, hits: list) -> str:
    qt = _qtokens(question)
    scored = []
    for rank, h in enumerate(hits[:3]):
        for s in _sentences(h.get("corrected_audio_text") or h.get("raw_audio_text", "")):
            st = set(tokens(s))
            scored.append((len(qt & st) / (len(qt) or 1) - 0.05 * rank, rank, s))
    scored.sort(key=lambda x: -x[0])
    picked = [x for x in scored[:3] if x[0] > 0]
    if not picked and hits:
        top = hits[0]
        txt = (top.get("corrected_audio_text") or top.get("raw_audio_text") or top.get("ocr_text") or "").strip()
        return txt[:400] if txt else ""
    picked.sort(key=lambda x: (x[1], x[2]))
    return " ".join(p[2] for p in picked)


def answer(index, llm, question: str, lecture_id: Optional[str] = None, top_k: int = 5) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "", "mode": "none", "sources": []}
    hits = rerank(question, index.query(question, top_k=max(top_k * 3, 10), lecture_id=lecture_id), top_k)
    if not hits:
        return {"answer": "لم أجد محاضرات مفهرسة للإجابة عن هذا السؤال.", "mode": "none", "sources": []}

    ctx = []
    for i, h in enumerate(hits[:4], 1):
        ctx.append(f"[{i}] {unit_label(h)}\nنص الشريحة: {h.get('ocr_text', '')[:500] or '-'}\n"
                   f"شرح المحاضر: {(h.get('corrected_audio_text') or h.get('raw_audio_text') or '-')[:900]}")
    text, mode = None, "extractive"
    if llm is not None and getattr(llm, "available", False):
        text = llm.chat(SYSTEM_QA, "المقاطع:\n" + "\n\n".join(ctx) + f"\n\nسؤال الطالب: {question}\nالإجابة:",
                        max_new_tokens=400)
        if text:
            mode = "llm"
    if not text:
        text = extractive_answer(question, hits)
        text = (f"بحسب المحاضرة: {text}" if text else "لم أجد إجابة واضحة؛ راجع المصادر المرفقة.")

    sources = [{
        "lecture_id": h["lecture_id"], "lecture_title": h.get("lecture_title", ""),
        "slide_id": h["slide_id"], "timestamp": h["timestamp"], "timestamp_str": fmt_time(h["timestamp"]),
        "slide_image_path": h.get("slide_image_path", ""),
        "snippet": (h.get("corrected_audio_text") or h.get("ocr_text") or "")[:240],
        "score": round(float(h["final"]), 4),
    } for h in hits]
    return {"answer": text, "mode": mode, "sources": sources}
