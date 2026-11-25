# 🎓 Lecture Engine — المساعد الجامعي الذكي (جامعة ذمار)

Pipeline متكامل: **رابط/فيديو ← رؤية (YOLOv8n-cls) + صوت (Whisper) ← دمج وتصحيح (Qwen) ← فهرسة (ChromaDB) ← سؤال وجواب (RAG)**
مع واجهة REST جاهزة لتطبيق الموبايل: رفع/رابط، متابعة الحالة، سؤال نصي/صوتي، وإجابة مع **رقم المحاضرة + رقم الشريحة + الدقيقة + صورة الشريحة**.

الفئات: `slide` · `blank` · `transition` فقط (لا توجد فئة `speaker`).

## 1) هيكلة المشروع

```
LectureEngine/
├── src/lecture_engine/         ← الـ Backend Pipeline (يُستخدم في التشغيل الفعلي)
│   ├── vision/                 frames · classifier(YOLOv8n-cls) · segmenter · similarity
│   ├── ocr.py                  PaddleOCR (2.x و3.x)         ├── asr.py     Whisper (faster-whisper)
│   ├── fusion.py               محاذاة زمنية + تصحيح Qwen    ├── llm.py     Qwen (local | api | off)
│   ├── index.py                ChromaDB + تضمين متعدد اللغات  ├── qa.py      RAG + الإجابة
│   ├── pipeline.py             ينسّق كل المراحل              ├── jobs.py    حالة المعالجة (SQLite)
│   ├── ingest.py               ملف/يوتيوب/Drive             └── api.py     واجهة FastAPI للموبايل
├── training/                   ← تجهيز البيانات وتدريب الرؤية (لا يدخل في التشغيل النهائي)
│   ├── prepare_data.py         إطار/2ث + حذف المتكرر + فرز أولي      ├── split_dataset.py  80/10/10 بلا تسرّب
│   ├── train_cls.py            ضبط YOLOv8n-cls + Top-1/Top-5/Confusion Matrix
│   └── run_training.py         كل ما سبق بأمر واحد
├── scripts/                    process_video.py · ask.py · run_server.sh · selftest.py
├── examples/                   client.py · curl_examples.sh
├── notebooks/LectureEngine_Colab_Training.ipynb   ← دفتر Colab
├── tests/                      اختبارات المنطق الأساسي
├── data/videos/  data/manual/{slide,blank,transition}/   ← مدخلاتك المحلية
├── models/best.pt              ← نموذج التصنيف المدرَّب (ناتج التدريب)
├── storage/                    ← يُنشأ: lectures/ + chroma/ + jobs.db
├── .env.example  requirements.txt  requirements_colab.txt
```

## 2) أين تضع الملفات؟

### على Google Drive (للتدريب في Colab)
```
MyDrive/UniversityAI/
├── LectureEngine.zip      ← هذا المشروع مضغوطاً
├── videos/                ← فيديوهات المحاضرات (3 على الأقل؛ الأفضل 8–10 بقاعات/دكاترة/إضاءة مختلفة)
│    └── videos.txt        ← (اختياري) روابط يوتيوب/درايف، سطر لكل رابط
├── manual/slide|blank|transition/   ← (اختياري) صور موسومة يدوياً لدعم الفئات الشحيحة
└── outputs/               ← يُنشأ تلقائياً: models/best.pt + reports/ + storage/
```

### على جهازك/الخادم (للتشغيل)
بعد التدريب: انسخ `outputs/models/best.pt` إلى `LectureEngine/models/best.pt`.
(وإن فهرست المحاضرات في Colab: انسخ `outputs/storage/` إلى `LectureEngine/storage/`.)

## 3) كيف تشغّل التدريب (Colab — الأسهل)

1. ارفع `LectureEngine.zip` والفيديوهات إلى `UniversityAI/` كما في الأعلى.
2. افتح `notebooks/LectureEngine_Colab_Training.ipynb` في Colab ← `Runtime ← Change runtime type ← T4 GPU`.
3. `Runtime ← Run all`. لا تغلق التبويب.
4. النتيجة في `UniversityAI/outputs/`: `models/best.pt` و`reports/` (`metrics.json`، `confusion_matrix_test.png/csv`، منحنيات التدريب).

**المراجعة اليدوية السريعة (موصى بها):** اضبط `STAGE="prepare"` ← يُنتج `UniversityAI/review/{slide,blank,transition}` ← افتحه في Drive وانقل أي صورة في مجلد خاطئ إلى مجلدها الصحيح أو احذفها (خصوصاً `transition`) ← اضبط `STAGE="train"` وأعد التشغيل. مراجعتك محفوظة بين الجلسات.

### أو محلياً (يفضّل GPU)
```bash
pip install -r requirements.txt
python training/run_training.py --videos data/videos --manual data/manual --out . 
#   --stage prepare   لإيقاف التنفيذ بعد الفرز الأولي (راجع work/review ثم أعد بـ --stage train)
#   --epochs 40       الافتراضي 30 على GPU و15 على CPU
```

### كيف يعمل التدريب ولماذا النتائج موثوقة
* **ضبط دقيق** لـ `yolov8n-cls.pt` (وزن جاهز) وليس تدريباً من الصفر.
* **منع التسرب:** ≥3 فيديوهات ← التقسيم **على مستوى الفيديو** (فيديو كامل لا يظهر في قسمين). أقل من 3 ← كتل زمنية متتالية بإطار فاصل (يُذكر ذلك في التقرير كتقييم أضعف).
* الصور اليدوية (`manual/`) والاصطناعية تدخل **التدريب فقط**.
* إن قلّت صور `blank`/`transition` الحقيقية (<40) تُضاف صور اصطناعية (شاشة سوداء/بيضاء، مزج وتمويه شرائح) للتدريب فقط، ويُذكر ذلك في `dataset_report.json`. وفئة `blank` تُصحَّح في التشغيل أيضاً بقاعدة سوداء/بيضاء دقيقة.
* **المخرجات:** Top-1، Top-5، Confusion Matrix (صورة + CSV + أرقام ونسب) لـ val وtest في `reports/` — للتوثيق والمناقشة.
* ⚠️ الفرز الأولي بقواعد خفيفة؛ لذلك المراجعة اليدوية ترفع الدقة الحقيقية. مع 3 فئات يكون Top-5 = 100% دائماً (مجرد تحقق).

## 4) تشغيل الخادم (واجهة الموبايل)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # عدّل ما تحتاج
python scripts/selftest.py       # فحص الجاهزية (✅/⚠️)
bash scripts/run_server.sh       # → http://localhost:8000/docs  (توثيق تفاعلي)
```
أو بدون خادم: `python scripts/process_video.py data/videos/lec1.mp4 --title "نظم التشغيل"` ثم `python scripts/ask.py "ما هي الذاكرة الافتراضية؟"`.

### Qwen — اختر خلفية
| الحالة | الإعداد |
|---|---|
| خادم بـ GPU | `QWEN_BACKEND=auto` (يحمّل `QWEN_MODEL` محلياً؛ 3B يكفي T4) |
| خادم بلا GPU | `QWEN_BACKEND=api` + `QWEN_API_BASE` + `QWEN_API_KEY` + `QWEN_API_MODEL` (أي خدمة متوافقة مع OpenAI، مثل DashScope أو Ollama) |
| بدون Qwen | `QWEN_BACKEND=off` → يستمر النظام بالتفريغ الخام، والإجابة باستخراج الجمل المطابقة مع مصدرها |

## 5) واجهة API لتطبيق الموبايل

أضف `X-API-Key` في كل طلب إن ضبطت `API_KEY` في `.env`. المسارات تحت `/api/v1`.

| الغرض | الطلب | الرد |
|---|---|---|
| رفع ملف | `POST /lectures` (multipart: `file`, `title`) | `202 {"lecture_id":"L001","status":"queued"}` |
| رابط يوتيوب/Drive | `POST /lectures` (form: `url`, `title`) | نفسه |
| متابعة الحالة | `GET /lectures/{id}/status` | `{"status":"processing","stage":"asr","progress":62.5,"message":"...","warnings":[],"error":""}` |
| القائمة / الشرائح / الحذف | `GET /lectures` · `GET /lectures/{id}/slides` · `DELETE /lectures/{id}` | |
| سؤال نصي | `POST /ask` JSON: `{"question","lecture_id"?,"top_k"?}` | انظر أدناه |
| سؤال صوتي | `POST /ask/voice` (multipart: `audio`, `lecture_id`?) | نفس رد `/ask` + `question` المفرَّغ |

`status`: `queued → processing → done | failed` (و`stage`: download, frames, classify, ocr, asr, fusion, index). `progress` من 0 إلى 100.

رد `/ask`:
```json
{
  "question": "ما هي الذاكرة الافتراضية؟",
  "answer": "الذاكرة الافتراضية تسمح بتشغيل برامج أكبر من الذاكرة الفعلية ... [1]",
  "mode": "llm",
  "sources": [{
    "lecture_id": "L001", "lecture_title": "نظم التشغيل 1", "slide_id": 12,
    "timestamp": 745.2, "timestamp_str": "12:25",
    "slide_image_url": "http://HOST:8000/media/L001/slides/slide_012.jpg",
    "video_url": "http://HOST:8000/media/L001/video.mp4",
    "snippet": "...", "score": 0.83
  }]
}
```
`mode`: `llm` (صياغة Qwen من الدليل فقط) | `extractive` (جمل من كلام المحاضر) | `none`. للقفز في الفيديو الأصلي استخدم `timestamp` (ثوانٍ).
أمثلة جاهزة: `examples/curl_examples.sh` و`examples/client.py`. اضبط `PUBLIC_BASE_URL` عند وضع الخادم خلف دومين/Proxy.

## 6) ماذا يحدث لكل فيديو

1. **جلب:** ملف أو يوتيوب (`yt-dlp`) أو Drive (`gdown`).
2. **رؤية:** إطار/2ث ← YOLOv8n-cls (`best.pt`) ← إبقاء `slide` ← دمج الإطارات المتشابهة (المؤشر والكتابة الخفيفة لا تقسم الشريحة) ← `start_time → end_time` وأوضح إطار ممثِّل.
3. **OCR:** PaddleOCR (عربي + إنجليزي) بعد تكبير وتحسين التباين، بترتيب قراءة صحيح لليمين/اليسار.
4. **صوت:** استخراج المسار الصوتي ← Whisper (`large-v3` على GPU) مع VAD وحذف الهلوسات ← `transcript.json/.srt`.
5. **دمج:** كل مقطع كلام يُربط بالشريحة الظاهرة وقتها ← Qwen يصحّح المصطلحات بمرجع نص الشريحة (مع حارس يرفض أي ناتج يلخّص/يغيّر المعنى).
6. **فهرسة:** وحدات `(lecture_id, slide_id, timestamp, slide_image_path, ocr_text, corrected_audio_text)` في ChromaDB (`multilingual-e5-small`).
7. **سؤال:** بحث دلالي + إعادة ترتيب معجمية ← إجابة Qwen أو استخراجية ← مصادر.

الاستئناف: إعادة المعالجة تستكمل من المراحل المحفوظة (شرائح/تفريغ) ما لم تستخدم `--force`، وإعادة تشغيل الخادم تستأنف الوظائف غير المنتهية.

## 7) أمان الأخطاء (لا انهيار)
* PaddleOCR فشل ← تستمر المعالجة بالصوت فقط وتظهر `warnings`.
* Whisper: GPU fp16 ← GPU int8 ← CPU تلقائياً. Qwen فشل ← النص الخام. نموذج التضمين تعذّر ← تضمين مجزّأ (hash).
* لا `best.pt` ← كشف الشرائح بقواعد خفيفة. التدريب: إعادة تلقائية بدفعة أصغر عند نفاد الذاكرة.
* الدفتر لا يرفع استثناءات: يطبع ✅/⚠️ ورمز خروج.

## 8) حدود يجب معرفتها
* تسجيلات تعرض **معاينة الشريحة التالية** بجانب الشريحة تُقرأ بـ OCR معها؛ إن ظهر ضجيج في `ocr_text` فاقتطاع منطقة الشريحة الرئيسية خطوة تحسين لاحقة.
* لا حماية للمسار `/media` (روابط الصور): ضعه خلف Proxy/Auth عند النشر العام.
* جودة الإجابة تتبع جودة التفريغ؛ للعربية استخدم `large-v3` على GPU.

## 9) الاختبار
```bash
pip install pytest && python -m pytest -q tests     # منطق الرؤية/التقسيم/الدمج/المحرّك
```

## 10) مشاكل شائعة
| العَرَض | الحل |
|---|---|
| Whisper: `libcudnn ... .so.8/.9` | `pip install -U ctranslate2` (أو `ctranslate2==4.4.0` لـ .so.8) — وإلا يعمل على CPU تلقائياً |
| الشريحة الواحدة تنقسم لعدة شرائح | ارفع `SLIDE_CHANGE_THRESHOLD` (مثلاً 0.01) |
| شرائح مختلفة تُدمج | اخفضها (مثلاً 0.002) |
| `transition` كثيرة خاطئة في review | صحّحها يدوياً ثم `STAGE=train` |
| `paddleocr` لا يُثبَّت | جرّب `pip install "paddlepaddle>=3.0" "paddleocr>=3.0"` ثم أعد المحاولة؛ النظام يعمل بدونه (بلا نص شرائح) |
