"""مثال عميل (يحاكي تطبيق الموبايل):  python examples/client.py video.mp4 "ما هو نظام التشغيل؟"
متغيرات: BASE_URL (افتراضي http://localhost:8000) و API_KEY إن ضُبط على الخادم."""
import os
import sys
import time

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
H = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

src, question = sys.argv[1], sys.argv[2]
if src.startswith("http"):
    r = requests.post(f"{BASE}/api/v1/lectures", data={"url": src, "title": "محاضرة"}, headers=H)
else:
    with open(src, "rb") as f:
        r = requests.post(f"{BASE}/api/v1/lectures", data={"title": os.path.basename(src)},
                          files={"file": f}, headers=H)
r.raise_for_status()
lid = r.json()["lecture_id"]
print("lecture:", lid)
while True:
    s = requests.get(f"{BASE}/api/v1/lectures/{lid}/status", headers=H).json()
    print(f"{s['status']:<11} {s['stage']:<9} {s['progress']:>5}%  {s['message']}")
    if s["status"] in ("done", "failed"):
        break
    time.sleep(5)
if s["status"] == "failed":
    sys.exit(s["error"])
a = requests.post(f"{BASE}/api/v1/ask", json={"question": question, "lecture_id": lid}, headers=H).json()
print("\nالإجابة:", a["answer"])
for x in a["sources"][:3]:
    print(f"  - {x['lecture_id']} | شريحة {x['slide_id']} | {x['timestamp_str']} | {x['slide_image_url']}")
