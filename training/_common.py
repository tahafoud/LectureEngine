import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ["slide", "blank", "transition"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
