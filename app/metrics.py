import threading
from statistics import mean
from metrics_logger import save_metrics


class MetricsCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.total_latencies = []
        self.retrieval_latencies = []
        self.reranking_latencies = []
        self.llm_latencies = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.tokens_per_second = []

    def record_request(self, latency_ms, success=True):
        with self.lock:
            self.request_count += 1
            self.total_latencies.append(latency_ms)
            if not success:
                self.error_count += 1

    def record_retrieval(self, latency_ms):
        with self.lock:
            self.retrieval_latencies.append(latency_ms)

    def record_reranking(self, latency_ms):
        with self.lock:
            self.reranking_latencies.append(latency_ms)

    def record_llm(self, latency_ms):
        with self.lock:
            self.llm_latencies.append(latency_ms)

    def record_tokens(self, prompt_tokens, completion_tokens, tokens_per_second):
        with self.lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            self.tokens_per_second.append(tokens_per_second)

    @staticmethod
    def percentile(values, percentile):
        if not values:
            return 0.0
        values = sorted(values)
        index = (len(values) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        fraction = index - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    def summary(self):
        with self.lock:
            total = self.request_count
            return {
                "requests": total,
                "errors": self.error_count,
                "error_rate_percent": (self.error_count / total) * 100 if total else 0.0,
                "total_latency_ms": {
                    "average": mean(self.total_latencies) if self.total_latencies else 0.0,
                    "p50": self.percentile(self.total_latencies, 0.50),
                    "p95": self.percentile(self.total_latencies, 0.95),
                },
                "retrieval_latency_ms": {"average": mean(self.retrieval_latencies) if self.retrieval_latencies else 0.0},
                "reranking_latency_ms": {"average": mean(self.reranking_latencies) if self.reranking_latencies else 0.0},
                "llm_latency_ms": {"average": mean(self.llm_latencies) if self.llm_latencies else 0.0},
                "llm_tokens": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "average_tokens_per_second": mean(self.tokens_per_second) if self.tokens_per_second else 0.0,
                },
            }

    def save_snapshot(self):
        save_metrics(self.summary())


metrics = MetricsCollector()
