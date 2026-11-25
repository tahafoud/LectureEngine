"""جلب الفيديو: ملف محلي أو رابط يوتيوب أو رابط Google Drive."""
from __future__ import annotations

import shutil
from pathlib import Path

from .utils import have_ffmpeg, is_url, log

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".3gp"}


def find_video(dest_dir: Path) -> "Path | None":
    for p in sorted(Path(dest_dir).glob("video.*")):
        if p.suffix.lower() in VIDEO_EXT:
            return p
    return None


def fetch_video(source: str, dest_dir: Path) -> Path:
    """ينزّل/ينسخ المصدر إلى dest_dir/video.<ext> ويعيد المسار."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = find_video(dest_dir)
    if existing:
        return existing

    if not is_url(source):
        src = Path(source).expanduser()
        if not src.exists():
            raise RuntimeError(f"ملف الفيديو غير موجود: {source}")
        out = dest_dir / ("video" + (src.suffix.lower() if src.suffix.lower() in VIDEO_EXT else ".mp4"))
        if src.resolve() != out.resolve():
            shutil.copy2(src, out)
        return out

    low = source.lower()
    if "drive.google.com" in low or "docs.google.com" in low:
        return _from_drive(source, dest_dir)
    return _from_ytdlp(source, dest_dir)


def _from_drive(url: str, dest_dir: Path) -> Path:
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("لتحميل روابط Drive ثبّت: pip install gdown") from e
    out = dest_dir / "video.mp4"
    log(f"[ingest] تحميل من Google Drive ...")
    res = gdown.download(url, str(out), quiet=True, fuzzy=True)
    if not res or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("تعذر تحميل رابط Drive. تأكد أن الرابط عام (Anyone with the link).")
    return out


def _from_ytdlp(url: str, dest_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("لتحميل روابط يوتيوب ثبّت: pip install yt-dlp") from e
    fmt = "bv*[height<=1080]+ba/b[height<=1080]/b" if have_ffmpeg() else "b[height<=1080]/b"
    opts = {
        "outtmpl": str(dest_dir / "video.%(ext)s"),
        "format": fmt,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }
    log("[ingest] تحميل الفيديو بواسطة yt-dlp ...")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"تعذر تحميل الرابط: {str(e)[:200]}") from e
    v = find_video(dest_dir)
    if not v:
        raise RuntimeError("لم يُنتج التحميل ملف فيديو.")
    return v
