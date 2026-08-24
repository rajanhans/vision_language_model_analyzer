"""Append-only, local run logging without image bytes or API credentials."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_run(question: str, result: dict[str, Any], log_dir: str | Path = "logs") -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "runs.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        **result,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination

