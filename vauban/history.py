"""Helpers to load and filter persisted campaign histories.

These operate on the JSON artifacts saved under ``reports/campaign_history_*.json``.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _latest_history_path(reports_dir: str = "reports") -> Optional[Path]:
    paths = sorted(Path(reports_dir).glob("campaign_history_*.json"))
    return paths[-1] if paths else None


def load_latest_history(reports_dir: str = "reports") -> List[Dict]:
    """Return the most recent campaign history; empty list when none exist."""
    path = _latest_history_path(reports_dir)
    if not path:
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def list_breaches(reports_dir: str = "reports") -> List[Dict]:
    """Return breached attacks from the latest run (is_breach True)."""
    hist = load_latest_history(reports_dir)
    return [r for r in hist if r.get("is_breach")]


def get_attack_by_id(attack_id: str, reports_dir: str = "reports") -> Optional[Dict]:
    """Lookup a single attack by id from the latest history artifact."""
    for entry in load_latest_history(reports_dir):
        if str(entry.get("id")) == str(attack_id):
            return entry
    return None
