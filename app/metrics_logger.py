import json
import os
from datetime import datetime, timezone
from threading import Lock

BASE_LOG_DIR = "data_logs"
_lock = Lock()


def _log_dir_for_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir = os.path.join(BASE_LOG_DIR, today)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_metrics(metrics_data):
    """Append the current CUMULATIVE aggregate snapshot.

    One compact JSON line per request to data_logs/<date>/metrics.jsonl.
    This is intentionally a rolling running-total (averages/percentiles
    across every request so far) -- good for plotting a trend over time,
    not for inspecting a single request.
    """
    try:
        log_dir = _log_dir_for_today()
        metrics_file = os.path.join(log_dir, "metrics.jsonl")
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics_data}
        with _lock, open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as e:
        print(f"Failed to save metrics: {e}")


def save_request_metrics(request_id, request_data):
    """Save ONE structured, pretty-printed JSON file per request.

    Written to data_logs/<date>/metrics/<request_id>.json with three
    clearly labelled sections: "request", "rag", "llm". This is the
    human-readable counterpart to the rolling metrics.jsonl above.
    """
    try:
        log_dir = _log_dir_for_today()
        metrics_dir = os.path.join(log_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        payload = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **request_data,
        }

        file_path = os.path.join(metrics_dir, f"{request_id}.json")
        with _lock, open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as e:
        print(f"Failed to save request metrics: {e}")
