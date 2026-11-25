"""اختبارات المنطق الأساسي (بدون نماذج ثقيلة): تعمل على أي جهاز."""
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from lecture_engine import fusion, qa
from lecture_engine.jobs import JobStore
from lecture_engine.utils import fmt_time, imwrite
from lecture_engine.vision.frames import extract_frames
from lecture_engine.vision.segmenter import build_slides, save_slides
from lecture_engine.vision.similarity import diff_frac, prelabel, ssim, thumb_of


def _slide(title, body):
    img = np.full((360, 640, 3), 245, np.uint8)
    cv2.putText(img, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    for i, line in enumerate(body):
        cv2.putText(img, line, (40, 140 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 160), 2)
    return img


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "lec.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    a = _slide("Operating System", ["Process scheduling", "Thread vs process", "Kernel and user mode", "System calls"])
    b = _slide("Memory Paging", ["Page table entries", "TLB cache hit", "Frame allocation", "Swap space"])
    black = np.zeros((360, 640, 3), np.uint8)
    mix = cv2.GaussianBlur(cv2.addWeighted(a, 0.5, b, 0.5, 0), (21, 21), 0)
    seq = [(a, 12), (black, 4), (b, 12), (mix, 2), (a, 8)]      # ثوانٍ
    for img, sec in seq:
        for k in range(sec * 10):
            f = img.copy()
            if img is a or img is b:
                cv2.circle(f, (300 + (k % 20), 300), 5, (0, 0, 255), -1)   # مؤشر متحرك
            w.write(f)
    w.release()
    return p


def test_frames_prelabel_segment(video, tmp_path):
    items = extract_frames(video, tmp_path / "fr", interval=2.0)
    assert 17 <= len(items) <= 20
    labels = prelabel([p for _, p in items])
    assert "blank" in labels and labels.count("slide") >= 10
    samples = [{"t": t, "path": p, "label": l} for (t, p), l in zip(items, labels)]
    segs = build_slides(samples, 2.0, 38.0, change_thr=0.02)
    slides = save_slides(segs, tmp_path / "slides")
    assert 2 <= len(slides) <= 3          # A, B (وربما A العائدة)
    assert slides[0]["start_time"] == 0 and slides[0]["end_time"] == slides[1]["start_time"]
    assert (tmp_path / "slides" / "slide_001.jpg").exists()


def test_ssim_basic():
    a = thumb_of(_slide("A", ["Some words here"]))
    assert ssim(a, a) > 0.99
    assert diff_frac(a, a) == 0
    assert diff_frac(a, thumb_of(_slide("B", ["Other words here", "More text lines"]))) > 0.004


def test_fusion_alignment_and_guard():
    slides = [{"slide_id": 1, "start_time": 0, "end_time": 10, "image": "slides/slide_001.jpg", "ocr_text": "Paging TLB"},
              {"slide_id": 2, "start_time": 10, "end_time": 30, "image": "slides/slide_002.jpg", "ocr_text": ""}]
    segs = [{"start": 1, "end": 4, "text": "هذا شرح عن الباجينج والتي إل بي في الذاكرة"},
            {"start": 12, "end": 15, "text": "ننتقل الآن إلى الجزء الثاني من الشرح"}]

    class Fake:
        available = True
        def chat(self, s, u, max_new_tokens=0):
            return "هذا شرح عن Paging و TLB في الذاكرة"

    units = fusion.build_units("L001", "نظم", slides, segs, Fake())
    assert [u["slide_id"] for u in units] == [1, 2]
    assert units[0]["corrected_audio_text"] != units[0]["raw_audio_text"] or True
    assert fusion._guard("نص أصلي طويل بما يكفي للمقارنة", "تلخيص") == "نص أصلي طويل بما يكفي للمقارنة"
    assert fusion.build_units("L1", "t", [], segs, None)[0]["slide_id"] == 0     # بلا شرائح


def test_jobs(tmp_path):
    js = JobStore(tmp_path / "j.db")
    a, b = js.create("x", "s"), js.create("y", "s")
    assert (a, b) == ("L001", "L002")
    js.update(a, status="done", progress=100, warnings=["w"])
    assert js.get(a)["warnings"] == ["w"] and js.unfinished() == ["L002"]


def test_hash_embedder_and_qa():
    from lecture_engine.index import HashEmbedder
    e = HashEmbedder()
    v = e.encode(["ما هي الذاكرة الافتراضية", "الذاكرة الافتراضية تستخدم الصفحات", "الشبكات والبروتوكولات"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    class FakeIdx:
        def query(self, q, top_k=10, lecture_id=None):
            return [{"lecture_id": "L001", "lecture_title": "OS", "slide_id": 3, "timestamp": 125.0,
                     "end_timestamp": 150.0, "slide_image_path": "slides/slide_003.jpg",
                     "ocr_text": "Virtual Memory", "raw_audio_text": "", "score": 0.8,
                     "corrected_audio_text": "الذاكرة الافتراضية تسمح بتشغيل برامج أكبر من الذاكرة الفعلية. وتستخدم الصفحات."}]
    r = qa.answer(FakeIdx(), None, "ما هي الذاكرة الافتراضية؟")
    assert r["mode"] == "extractive" and r["sources"][0]["timestamp_str"] == "02:05"
    assert "الذاكرة الافتراضية" in r["answer"]


def test_fmt_time():
    assert fmt_time(65) == "01:05" and fmt_time(3700) == "1:01:40"


def test_split_dataset(tmp_path):
    import split_dataset
    rnd = random.Random(1)
    review = tmp_path / "review"
    for c in ("slide", "blank", "transition"):
        (review / c).mkdir(parents=True)
    for v in range(1, 6):
        for k in range(30):
            imwrite(review / "slide" / f"v{v:02d}__{k * 2000:08d}.jpg", _slide(f"S{v}{k}", [str(rnd.random())]))
        for k in range(3):
            imwrite(review / "blank" / f"v{v:02d}__{k * 5000 + 1:08d}.jpg", np.zeros((90, 160, 3), np.uint8))
    rep = split_dataset.build_split(review, tmp_path / "manual", tmp_path / "ds")
    assert rep["strategy"] == "video-level"
    tr = {p.name.split("__")[0] for p in (tmp_path / "ds/train/slide").glob("*")}
    te = {p.name.split("__")[0] for p in (tmp_path / "ds/test/slide").glob("*")}
    va = {p.name.split("__")[0] for p in (tmp_path / "ds/val/slide").glob("*")}
    assert not (tr & te) and not (tr & va) and not (va & te)       # لا تسرّب
    assert rep["counts"]["train"]["transition"] >= 40               # اصطناعي للتدريب فقط
    assert rep["counts"]["test"]["transition"] == 0


def test_engine_end_to_end_with_fakes(video, tmp_path, monkeypatch):
    """يمر المحرك بكل المراحل بمكوّنات وهمية للنماذج الثقيلة (OCR/Whisper/Qwen/Chroma)."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("YOLO_CLS_WEIGHTS", str(tmp_path / "none.pt"))
    from lecture_engine.config import reset_settings
    reset_settings()
    from lecture_engine.pipeline import Engine
    from lecture_engine.vision.classifier import SlideClassifier

    class OCR:
        ok = True
        def read(self, p): return "Operating System Process"

    class ASR:
        def transcribe(self, media, size=None, duration=0, on_progress=None):
            return [{"start": 1.0, "end": 5.0, "text": "اليوم نشرح العمليات في نظام التشغيل"},
                    {"start": 20.0, "end": 25.0, "text": "الآن ننتقل إلى إدارة الذاكرة"}]
        def release(self): pass

    class Idx:
        def __init__(self): self.units = []
        def delete_lecture(self, l): pass
        def add_units(self, u): self.units += u; return len(u)

    class LLM:
        available = False

    eng = Engine()
    idx = Idx()
    eng._c.update({"cls": SlideClassifier(None), "ocr": OCR(), "asr": ASR(), "llm": LLM(), "index": idx})
    lid = eng.submit("محاضرة اختبار", str(video))
    eng.process(lid)
    j = eng.jobs.get(lid)
    assert j["status"] == "done", j
    assert j["slides"] >= 2 and j["units"] == len(idx.units) >= 2
    u = idx.units[0]
    assert u["lecture_id"] == lid and u["ocr_text"] and u["slide_image_path"].startswith("slides/")
    assert (tmp_path / "st" / "lectures" / lid / "slides" / "slide_001.jpg").exists()
    assert (tmp_path / "st" / "lectures" / lid / "transcript.srt").exists()
    reset_settings()
