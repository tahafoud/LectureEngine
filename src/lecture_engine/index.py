"""ChromaDB: فهرسة وحدات المحاضرات والبحث الدلالي (تضمينات متعددة اللغات)."""
from __future__ import annotations

import threading
import zlib
from typing import Optional

import numpy as np

from .config import Settings
from .utils import has_cuda, log, normalize_ar


class HashEmbedder:
    """بديل بلا تنزيل: n-grams حرفية مجزّأة (hashing). أضعف دلالياً لكنه لا يفشل أبداً."""
    tag = "hash"
    DIM = 768

    def encode(self, texts: list, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            t = normalize_ar(t)
            words = t.split()
            feats = list(words)
            for w in words:
                w2 = f"#{w}#"
                for n in (3, 4):
                    feats += [w2[j:j + n] for j in range(max(1, len(w2) - n + 1))]
            for f in feats:
                out[i, zlib.crc32(f.encode("utf-8")) % self.DIM] += 1.0
            nrm = np.linalg.norm(out[i])
            if nrm:
                out[i] /= nrm
        return out


class STEmbedder:
    tag = "e5"

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cuda" if has_cuda() else "cpu")
        self.e5 = "e5" in name.lower()
        self.tag = "st_" + "".join(c for c in name.split("/")[-1].lower() if c.isalnum())[:24]

    def encode(self, texts: list, is_query: bool = False) -> np.ndarray:
        if self.e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def make_embedder(s: Settings):
    if s.embed_backend != "hash":
        try:
            e = STEmbedder(s.embed_model)
            log(f"[index] نموذج التضمين: {s.embed_model}")
            return e
        except Exception as ex:  # noqa: BLE001
            log(f"[index] تعذر تحميل نموذج التضمين ({str(ex)[:100]})؛ سيُستخدم التضمين المجزّأ (hash).")
    return HashEmbedder()


def embed_text(u: dict) -> str:
    return (u.get("ocr_text", "")[:600] + "\n" + (u.get("corrected_audio_text") or u.get("raw_audio_text", ""))[:900]).strip()


class LectureIndex:
    def __init__(self, s: Settings, embedder=None):
        import chromadb
        self.embedder = embedder or make_embedder(s)
        s.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(s.chroma_dir))
        name = f"lectures_{self.embedder.tag}"
        try:
            self.col = self.client.get_or_create_collection(
                name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        except Exception:  # noqa: BLE001 - إصدارات أقدم
            self.col = self.client.get_or_create_collection(
                name=name, embedding_function=None, metadata={"hnsw:space": "cosine"})
        self._lock = threading.Lock()

    def count(self) -> int:
        return self.col.count()

    def delete_lecture(self, lecture_id: str) -> None:
        with self._lock:
            try:
                self.col.delete(where={"lecture_id": lecture_id})
            except Exception:  # noqa: BLE001
                pass

    def add_units(self, units: list) -> int:
        if not units:
            return 0
        docs = [embed_text(u) or "-" for u in units]
        embs = self.embedder.encode(docs).tolist()
        metas = [{
            "lecture_id": u["lecture_id"], "lecture_title": u["lecture_title"], "slide_id": int(u["slide_id"]),
            "timestamp": float(u["timestamp"]), "end_timestamp": float(u["end_timestamp"]),
            "slide_image_path": u["slide_image_path"], "ocr_text": u["ocr_text"],
            "corrected_audio_text": u["corrected_audio_text"], "raw_audio_text": u["raw_audio_text"],
        } for u in units]
        ids = [u["unit_id"] for u in units]
        with self._lock:
            for i in range(0, len(ids), 200):
                self.col.upsert(ids=ids[i:i + 200], embeddings=embs[i:i + 200],
                                documents=docs[i:i + 200], metadatas=metas[i:i + 200])
        return len(ids)

    def query(self, question: str, top_k: int = 10, lecture_id: Optional[str] = None) -> list:
        n = self.count()
        if n == 0:
            return []
        q = self.embedder.encode([question], is_query=True)[0]
        kw = dict(query_embeddings=[q.tolist()], n_results=min(top_k, n),
                  include=["metadatas", "embeddings"])
        if lecture_id:
            kw["where"] = {"lecture_id": lecture_id}
        with self._lock:
            try:
                r = self.col.query(**kw)
            except Exception as e:  # noqa: BLE001
                log(f"[index] خطأ بحث: {str(e)[:100]}")
                return []
        hits = []
        metas = (r.get("metadatas") or [[]])[0]
        embs = r.get("embeddings")
        embs = embs[0] if embs is not None and len(embs) else []
        for i, m in enumerate(metas):
            score = float(np.dot(q, np.asarray(embs[i], dtype=np.float32))) if i < len(embs) else 0.0
            hits.append({**m, "score": score})
        return hits
