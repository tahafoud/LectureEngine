"""الإعدادات: تُقرأ من متغيرات البيئة أو ملف .env في جذر المشروع."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    f = Path(os.environ.get("ENV_FILE", str(ROOT / ".env")))
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _s(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


def _i(name: str, default: int) -> int:
    try:
        return int(_s(name, str(default)))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(_s(name, str(default)))
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    p = Path(_s(name, str(default))).expanduser()
    return p if p.is_absolute() else (ROOT / p)


@dataclass
class Settings:
    storage_dir: Path
    yolo_weights: Path
    frame_interval: float
    slide_change_threshold: float
    min_slide_frames: int
    ocr_langs: list
    ocr_min_score: float
    asr_model: str          # auto | large-v3 | medium | small ...
    asr_query_model: str    # نموذج الأسئلة الصوتية (أصغر)
    asr_language: str       # ar | en | auto
    qwen_backend: str       # auto | local | api | off
    qwen_model: str
    qwen_api_base: str
    qwen_api_key: str
    qwen_api_model: str
    embed_backend: str      # auto | st | hash
    embed_model: str
    api_key: str
    max_upload_mb: int
    public_base_url: str
    top_k: int
    cors_origins: list

    @property
    def lectures_dir(self) -> Path:
        return self.storage_dir / "lectures"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "jobs.db"


def _build() -> Settings:
    _load_dotenv()
    storage = _path("STORAGE_DIR", ROOT / "storage")
    return Settings(
        storage_dir=storage,
        yolo_weights=_path("YOLO_CLS_WEIGHTS", ROOT / "models" / "best.pt"),
        frame_interval=_f("FRAME_INTERVAL", 2.0),
        slide_change_threshold=_f("SLIDE_CHANGE_THRESHOLD", 0.004),
        min_slide_frames=_i("MIN_SLIDE_FRAMES", 2),
        ocr_langs=[x.strip() for x in _s("OCR_LANGS", "ar,en").split(",") if x.strip()],
        ocr_min_score=_f("OCR_MIN_SCORE", 0.5),
        asr_model=_s("ASR_MODEL", "auto"),
        asr_query_model=_s("ASR_QUERY_MODEL", "small"),
        asr_language=_s("ASR_LANGUAGE", "ar"),
        qwen_backend=_s("QWEN_BACKEND", "auto").lower(),
        qwen_model=_s("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        qwen_api_base=_s("QWEN_API_BASE", ""),
        qwen_api_key=_s("QWEN_API_KEY", ""),
        qwen_api_model=_s("QWEN_API_MODEL", "qwen-plus"),
        embed_backend=_s("EMBED_BACKEND", "auto").lower(),
        embed_model=_s("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        api_key=_s("API_KEY", ""),
        max_upload_mb=_i("MAX_UPLOAD_MB", 4096),
        public_base_url=_s("PUBLIC_BASE_URL", "").rstrip("/"),
        top_k=_i("TOP_K", 5),
        cors_origins=[x.strip() for x in _s("CORS_ORIGINS", "*").split(",") if x.strip()],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _build()


def reset_settings() -> None:
    get_settings.cache_clear()
