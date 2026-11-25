"""واجهة REST لتطبيق الموبايل (FastAPI).

تشغيل:  uvicorn lecture_engine.api:app --host 0.0.0.0 --port 8000
التوثيق التفاعلي: http://localhost:8000/docs
"""
from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import VIDEO_EXT
from .pipeline import Engine
from .utils import is_url, log, read_json


class AskIn(BaseModel):
    question: str = Field(..., min_length=2, description="سؤال الطالب")
    lecture_id: Optional[str] = Field(None, description="اختياري: حصر البحث في محاضرة واحدة")
    top_k: int = Field(5, ge=1, le=10)


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    s = get_settings()
    eng = engine or Engine(s)
    jobs_q: "queue.Queue[str]" = queue.Queue()

    def worker() -> None:
        while True:
            lid = jobs_q.get()
            try:
                eng.process(lid)
            except Exception as e:  # noqa: BLE001
                log(f"[worker] {lid}: {e}")
            finally:
                jobs_q.task_done()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=worker, daemon=True).start()
        for lid in eng.jobs.unfinished():       # استئناف ما انقطع عند إعادة التشغيل
            jobs_q.put(lid)
        yield

    app = FastAPI(title="Lecture Engine API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    s.lectures_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(s.lectures_dir)), name="media")

    def auth(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
        if s.api_key and x_api_key != s.api_key:
            raise HTTPException(401, "مفتاح API غير صحيح (X-API-Key)")

    def base(req: Request) -> str:
        return s.public_base_url or str(req.base_url).rstrip("/")

    def media_url(req: Request, lid: str, rel: str) -> Optional[str]:
        return f"{base(req)}/media/{lid}/{rel}" if rel else None

    def job_view(j: dict) -> dict:
        return {k: j[k] for k in ("lecture_id", "title", "status", "stage", "progress", "message",
                                  "error", "warnings", "slides", "units", "duration", "created", "updated")}

    # ------------------------------------------------------------------ صحة
    @app.get("/health")
    def health():
        return {"status": "ok", "queue": jobs_q.qsize()}

    # ------------------------------------------------------------------ رفع/رابط
    @app.post("/api/v1/lectures", status_code=202, dependencies=[Depends(auth)])
    def create_lecture(title: str = Form(""), url: str = Form(""), file: Optional[UploadFile] = File(None)):
        url = (url or "").strip()
        has_file = file is not None and bool(file.filename)
        if not has_file and not url:
            raise HTTPException(400, "أرسل ملف فيديو (file) أو رابطاً (url).")
        if has_file and url:
            raise HTTPException(400, "أرسل ملفاً أو رابطاً وليس الاثنين معاً.")
        if url and not is_url(url):
            raise HTTPException(400, "الرابط يجب أن يبدأ بـ http:// أو https://")
        if has_file:
            ext = Path(file.filename).suffix.lower()
            if ext not in VIDEO_EXT:
                raise HTTPException(400, f"صيغة غير مدعومة: {ext or 'بدون امتداد'}")
            title = title or Path(file.filename).stem
            lid = eng.submit(title, file.filename, "file")
            dest = s.lectures_dir / lid / f"video{ext}"
            limit, size = s.max_upload_mb * 1024 * 1024, 0
            try:
                with open(dest, "wb") as out:
                    while True:
                        chunk = file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(413, f"الملف أكبر من {s.max_upload_mb}MB")
                        out.write(chunk)
            except HTTPException:
                eng.delete(lid)
                raise
            except Exception as e:  # noqa: BLE001
                eng.delete(lid)
                raise HTTPException(500, f"فشل حفظ الملف: {e}")
        else:
            lid = eng.submit(title or url, url, "url")
        jobs_q.put(lid)
        return {"lecture_id": lid, "status": "queued"}

    # ------------------------------------------------------------------ الحالة
    @app.get("/api/v1/lectures", dependencies=[Depends(auth)])
    def list_lectures():
        return [job_view(j) for j in eng.jobs.list()]

    @app.get("/api/v1/lectures/{lid}", dependencies=[Depends(auth)])
    @app.get("/api/v1/lectures/{lid}/status", dependencies=[Depends(auth)])
    def lecture_status(lid: str):
        j = eng.jobs.get(lid)
        if not j:
            raise HTTPException(404, "محاضرة غير موجودة")
        return job_view(j)

    @app.get("/api/v1/lectures/{lid}/slides", dependencies=[Depends(auth)])
    def lecture_slides(lid: str, request: Request):
        if not eng.jobs.get(lid):
            raise HTTPException(404, "محاضرة غير موجودة")
        sl = read_json(s.lectures_dir / lid / "slides.json", []) or []
        return [{"slide_id": x["slide_id"], "start_time": x["start_time"], "end_time": x["end_time"],
                 "ocr_text": x.get("ocr_text", ""), "image_url": media_url(request, lid, x["image"])} for x in sl]

    @app.delete("/api/v1/lectures/{lid}", dependencies=[Depends(auth)])
    def delete_lecture(lid: str):
        if not eng.jobs.get(lid):
            raise HTTPException(404, "محاضرة غير موجودة")
        eng.delete(lid)
        return {"deleted": lid}

    # ------------------------------------------------------------------ الأسئلة
    def _ask(request: Request, question: str, lecture_id: Optional[str], top_k: int) -> dict:
        if lecture_id:
            j = eng.jobs.get(lecture_id)
            if not j:
                raise HTTPException(404, "محاضرة غير موجودة")
            if j["status"] != "done":
                raise HTTPException(409, f"المحاضرة لم تكتمل معالجتها بعد (الحالة: {j['status']})")
        res = eng.ask(question, lecture_id, top_k)
        for src in res["sources"]:
            src["slide_image_url"] = media_url(request, src["lecture_id"], src.pop("slide_image_path", ""))
            src["video_url"] = f"{base(request)}/media/{src['lecture_id']}/video.mp4"
        res["question"] = question
        return res

    @app.post("/api/v1/ask", dependencies=[Depends(auth)])
    def ask(body: AskIn, request: Request):
        return _ask(request, body.question, body.lecture_id, body.top_k)

    @app.post("/api/v1/ask/voice", dependencies=[Depends(auth)])
    def ask_voice(request: Request, audio: UploadFile = File(...), lecture_id: str = Form(""),
                  top_k: int = Form(5)):
        suffix = Path(audio.filename or "q.m4a").suffix or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            path = Path(tmp.name)
        try:
            q = eng.transcribe_question(path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"تعذر تفريغ السؤال الصوتي: {str(e)[:150]}")
        finally:
            path.unlink(missing_ok=True)
        if not q:
            raise HTTPException(422, "لم يُسمع كلام واضح في التسجيل.")
        return _ask(request, q, lecture_id or None, max(1, min(top_k, 10)))

    return app


app = create_app()
