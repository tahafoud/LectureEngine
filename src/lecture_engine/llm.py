"""Qwen: تصحيح سياقي لتفريغ الصوت + توليد الإجابات. خلفيات: local (transformers) | api (متوافق مع OpenAI) | off."""
from __future__ import annotations

from typing import Optional

from .config import Settings
from .utils import has_cuda, log


class QwenLLM:
    def __init__(self, s: Settings):
        self.s = s
        b = s.qwen_backend
        if b == "auto":
            b = "local" if has_cuda() else ("api" if s.qwen_api_base and s.qwen_api_key else "off")
        elif b == "api" and not (s.qwen_api_base and s.qwen_api_key):
            log("[llm] QWEN_BACKEND=api لكن QWEN_API_BASE/QWEN_API_KEY غير مضبوطين؛ تم التعطيل.")
            b = "off"
        self.backend = b
        self._tok = None
        self._model = None
        self._dead = False
        log(f"[llm] خلفية Qwen: {self.backend}")

    @property
    def available(self) -> bool:
        return self.backend in ("local", "api") and not self._dead

    # ------------------------------------------------------------------ local
    def _load_local(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = self.s.qwen_model
        log(f"[llm] تحميل {name} ...")
        self._tok = AutoTokenizer.from_pretrained(name)
        kw = dict(torch_dtype=torch.float16 if has_cuda() else torch.float32)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **kw)
        except Exception:  # noqa: BLE001 - accelerate غير موجود
            self._model = AutoModelForCausalLM.from_pretrained(name, **kw)
            self._model.to("cuda" if has_cuda() else "cpu")
        self._model.eval()

    def _chat_local(self, system: str, user: str, max_new_tokens: int) -> str:
        import torch
        self._load_local()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       repetition_penalty=1.05)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ api
    def _chat_api(self, system: str, user: str, max_new_tokens: int) -> str:
        import requests
        r = requests.post(
            self.s.qwen_api_base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.s.qwen_api_key}"},
            json={"model": self.s.qwen_api_model, "temperature": 0.0, "max_tokens": max_new_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> Optional[str]:
        """يعيد None عند أي فشل (لا يرفع استثناء) ليستمر النظام بالنص الخام."""
        if not self.available:
            return None
        try:
            return self._chat_local(system, user, max_new_tokens) if self.backend == "local" \
                else self._chat_api(system, user, max_new_tokens)
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:150]
            log(f"[llm] خطأ: {msg}")
            if self.backend == "local":     # لا نكرر محاولات فاشلة (ذاكرة/تنزيل)
                self._dead = True
                log("[llm] تم تعطيل Qwen لهذه الجلسة؛ سيُستخدم النص الخام/الاستخراجي.")
            return None

    def release(self) -> None:
        self._model = None
        self._tok = None
        from .utils import free_gpu
        free_gpu()
