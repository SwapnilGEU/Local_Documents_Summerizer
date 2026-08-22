import json
import os
from datetime import datetime
from threading import Lock

BASE_LOG_DIR = "data_logs"
_lock = Lock()


def save_metrics(metrics_data):
    """Save a metrics snapshot to data_logs/YYYY-MM-DD/metrics.jsonl."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(BASE_LOG_DIR, today)
        os.makedirs(log_dir, exist_ok=True)
        metrics_file = os.path.join(log_dir, "metrics.jsonl")
        payload = {"timestamp": datetime.now().isoformat(), **metrics_data}
        with _lock:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to save metrics: {e}")
