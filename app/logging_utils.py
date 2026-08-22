import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone


BASE_LOG_DIR = "data_logs"


class JsonFormatter(logging.Formatter):

    def format(self, record):

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(
                record,
                "event",
                record.getMessage()
            ),
        }

        fields = getattr(
            record,
            "fields",
            None
        )

        if fields:
            payload.update(fields)

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            payload,
            default=str
        )


logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

if not logger.handlers:

    # -----------------------------
    # Console logging (still one line per event, for live tailing)
    # -----------------------------

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setFormatter(
        JsonFormatter()
    )

    logger.addHandler(
        console_handler
    )

    logger.propagate = False


# -----------------------------
# Per-request buffering for file persistence
#
# Instead of writing one flat line per event, we collect every event for a
# given request_id in memory and only write the file once, as a single
# nested object, when the request finishes (request_completed / failed).
# -----------------------------

_buffer_lock = threading.Lock()
_request_buffers = {}


def _write_request_log(request_id):

    with _buffer_lock:
        record = _request_buffers.pop(request_id, None)

    if record is None:
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir = os.path.join(BASE_LOG_DIR, today)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.jsonl")

    with open(log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def log_event(
    event: str,
    **fields
):

    logger.info(
        event,
        extra={
            "event": event,
            "fields": fields
        }
    )

    request_id = fields.get("request_id")

    if request_id is None:
        return

    event_entry = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in fields.items() if k not in ("request_id", "query")},
    }

    with _buffer_lock:
        record = _request_buffers.setdefault(
            request_id,
            {
                "query": fields.get("query"),
                "request_id": request_id,
                "events": [],
            },
        )
        record["events"].append(event_entry)

    if event in ("request_completed", "request_failed"):
        _write_request_log(request_id)