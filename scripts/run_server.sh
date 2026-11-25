#!/usr/bin/env bash
# تشغيل الخادم:  bash scripts/run_server.sh
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
exec uvicorn lecture_engine.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
