#!/usr/bin/env bash
# أمثلة اختبار بـ curl (استبدل العنوان؛ أضف -H "X-API-Key: ..." إن ضبطت API_KEY)
BASE=${BASE:-http://localhost:8000}

curl $BASE/health

# 1) رفع ملف فيديو
curl -X POST $BASE/api/v1/lectures -F "title=نظم التشغيل 1" -F "file=@lecture1.mp4"
# 1-ب) أو رابط يوتيوب / Drive
curl -X POST $BASE/api/v1/lectures -F "title=محاضرة" -F "url=https://youtu.be/XXXXXXXX"

# 2) متابعة الحالة (queued → processing → done | failed)
curl $BASE/api/v1/lectures/L001/status

# 3) سؤال نصي (lecture_id اختياري)
curl -X POST $BASE/api/v1/ask -H "Content-Type: application/json" \
     -d '{"question":"ما هي الذاكرة الافتراضية؟","lecture_id":"L001","top_k":5}'

# 4) سؤال صوتي
curl -X POST $BASE/api/v1/ask/voice -F "audio=@question.m4a" -F "lecture_id=L001"

# 5) شرائح المحاضرة / قائمة المحاضرات / الحذف
curl $BASE/api/v1/lectures/L001/slides
curl $BASE/api/v1/lectures
curl -X DELETE $BASE/api/v1/lectures/L001
