"""تتبع حالة معالجة المحاضرات في SQLite (خفيف ولا يعطّل التدفق الأساسي)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

_COLS = ("lecture_id", "title", "source", "kind", "status", "stage", "progress", "message",
         "error", "warnings", "slides", "units", "duration", "created", "updated")


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS lectures(
                id INTEGER PRIMARY KEY AUTOINCREMENT, lecture_id TEXT UNIQUE, title TEXT, source TEXT,
                kind TEXT, status TEXT, stage TEXT, progress REAL DEFAULT 0, message TEXT DEFAULT '',
                error TEXT DEFAULT '', warnings TEXT DEFAULT '[]', slides INTEGER DEFAULT 0,
                units INTEGER DEFAULT 0, duration REAL DEFAULT 0, created REAL, updated REAL)""")

    def _conn(self):
        c = sqlite3.connect(str(self.path), timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def create(self, title: str, source: str, kind: str = "file") -> str:
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO lectures(lecture_id,title,source,kind,status,stage,progress,created,updated)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, title, source, kind, "queued", "queued", 0.0, now, now))
            lid = f"L{cur.lastrowid:03d}"
            c.execute("UPDATE lectures SET lecture_id=? WHERE id=?", (lid, cur.lastrowid))
        return lid

    def _row(self, r) -> dict:
        d = {k: r[k] for k in _COLS}
        try:
            d["warnings"] = json.loads(d["warnings"] or "[]")
        except json.JSONDecodeError:
            d["warnings"] = []
        return d

    def get(self, lecture_id: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM lectures WHERE lecture_id=?", (lecture_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit: int = 200) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM lectures ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def update(self, lecture_id: str, **fields) -> None:
        if not fields:
            return
        if "warnings" in fields and not isinstance(fields["warnings"], str):
            fields["warnings"] = json.dumps(fields["warnings"], ensure_ascii=False)
        fields["updated"] = time.time()
        keys = [k for k in fields if k in _COLS]
        with self._conn() as c:
            c.execute(f"UPDATE lectures SET {','.join(k + '=?' for k in keys)} WHERE lecture_id=?",
                      [fields[k] for k in keys] + [lecture_id])

    def unfinished(self) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT lecture_id FROM lectures WHERE status IN ('queued','processing')"
                             " ORDER BY id").fetchall()
        return [r["lecture_id"] for r in rows]

    def delete(self, lecture_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM lectures WHERE lecture_id=?", (lecture_id,))
