import json
import logging
import os
import sys
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


class DailyJsonFileHandler(logging.Handler):

    def emit(self, record):

        try:

            # Current date
            today = datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d")

            # data_logs/YYYY-MM-DD
            log_dir = os.path.join(
                BASE_LOG_DIR,
                today
            )

            os.makedirs(
                log_dir,
                exist_ok=True
            )

            # data_logs/YYYY-MM-DD/app.jsonl
            log_file = os.path.join(
                log_dir,
                "app.jsonl"
            )

            message = self.format(record)

            with open(
                log_file,
                "a",
                encoding="utf-8"
            ) as file:

                file.write(
                    message + "\n"
                )

        except Exception:

            self.handleError(record)


logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

if not logger.handlers:

    # -----------------------------
    # Console logging
    # -----------------------------

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setFormatter(
        JsonFormatter()
    )

    # -----------------------------
    # Persistent file logging
    # -----------------------------

    file_handler = DailyJsonFileHandler()

    file_handler.setFormatter(
        JsonFormatter()
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    logger.propagate = False


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