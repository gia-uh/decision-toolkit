"""Small, dependency-free Prometheus text exposition.

The server owns a ``Metrics`` instance; the core stays provider-agnostic.  Labels
are deliberately low-cardinality: HTTP paths are normalized to route templates,
and state/question content is never used as a label.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping

_DEFAULT_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

_HELP = {
    "dtk_http_requests_total": ("counter", "HTTP requests by method, route and status."),
    "dtk_http_request_duration_seconds": ("histogram", "HTTP request duration in seconds."),
    "dtk_decisions_total": ("counter", "Decision requests by model, variant and status."),
    "dtk_decision_duration_seconds": ("histogram", "Decision inference duration in seconds."),
    "dtk_input_tokens_total": ("counter", "Input tokens reported by the core."),
    "dtk_output_tokens_total": ("counter", "Output tokens reported by the core."),
    "dtk_model_load_seconds": ("histogram", "Model load/warmup duration in seconds."),
    "dtk_rate_limited_total": ("counter", "Requests rejected by the rate limiter."),
    "dtk_server_busy_total": ("counter", "Requests rejected because the inference gate was full."),
    "dtk_inflight_requests": ("gauge", "HTTP requests currently in flight."),
    "dtk_inference_inflight": ("gauge", "Inference calls currently holding the concurrency gate."),
    "dtk_model_cache_entries": ("gauge", "Models currently loaded in the registry."),
}

_KNOWN_PATHS = {
    "/health",
    "/v1/health",
    "/ready",
    "/v1/ready",
    "/v1/info",
    "/v1/models",
    "/v1/models/{id}",
    "/v1/decision-models",
    "/v1/systemone",
    "/v1/decisions",
    "/metrics",
}


def metric_path(path: str) -> str:
    """Collapse high-cardinality paths into stable route labels."""
    if path.startswith("/v1/models/"):
        return "/v1/models/{id}"
    if path in _KNOWN_PATHS:
        return path
    if path.startswith(("/docs", "/redoc")) or path == "/openapi.json":
        return "docs"
    return "other"


def _labels_key(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: tuple[tuple[str, str], ...], extra: tuple[tuple[str, str], ...] = ()) -> str:
    merged = dict(labels)
    merged.update(dict(extra))
    if not merged:
        return ""
    body = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(merged.items()))
    return "{" + body + "}"


def _number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return format(value, ".17g")


class Metrics:
    """Thread-safe counters, gauges and histograms with Prometheus rendering."""

    def __init__(self, buckets: Iterable[float] = _DEFAULT_BUCKETS) -> None:
        self._lock = threading.RLock()
        self._buckets = tuple(sorted({float(value) for value in buckets}))
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            list[float],
        ] = {}

    def inc(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        value: float = 1.0,
    ) -> None:
        with self._lock:
            self._counters[(name, _labels_key(labels))] += float(value)

    def add_gauge(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        value: float = 1.0,
    ) -> None:
        with self._lock:
            self._gauges[(name, _labels_key(labels))] += float(value)

    def set_gauge(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        value: float = 0.0,
    ) -> None:
        with self._lock:
            self._gauges[(name, _labels_key(labels))] = float(value)

    def observe(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        value: float = 0.0,
    ) -> None:
        key = (name, _labels_key(labels))
        with self._lock:
            buckets = self._histograms.get(key)
            if buckets is None:
                buckets = [0.0] * (len(self._buckets) + 2)  # bucket counts, sum, count
                self._histograms[key] = buckets
            observed = float(value)
            for index, boundary in enumerate(self._buckets):
                if observed <= boundary:
                    buckets[index] += 1.0
            buckets[-2] += observed
            buckets[-1] += 1.0

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {key: list(value) for key, value in self._histograms.items()}

        lines: list[str] = []
        for name in sorted({key[0] for key in counters}):
            kind, help_text = _HELP.get(name, ("counter", name))
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"))
            for (counter_name, labels), value in sorted(counters.items()):
                if counter_name != name:
                    continue
                lines.append(f"{name}{_format_labels(labels)} {_number(value)}")
        for name in sorted({key[0] for key in gauges}):
            kind, help_text = _HELP.get(name, ("gauge", name))
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"))
            for (gauge_name, labels), value in sorted(gauges.items()):
                if gauge_name != name:
                    continue
                lines.append(f"{name}{_format_labels(labels)} {_number(value)}")
        for name in sorted({key[0] for key in histograms}):
            kind, help_text = _HELP.get(name, ("histogram", name))
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"))
            for (histogram_name, labels), buckets in sorted(histograms.items()):
                if histogram_name != name:
                    continue
                for index, boundary in enumerate(self._buckets):
                    lines.append(
                        f"{name}_bucket"
                        f"{_format_labels(labels, (('le', _number(boundary)),))} "
                        f"{_number(buckets[index])}"
                    )
                lines.append(
                    f"{name}_bucket{_format_labels(labels, (('le', '+Inf'),))} "
                    f"{_number(buckets[-1])}"
                )
                lines.append(f"{name}_sum{_format_labels(labels)} {_number(buckets[-2])}")
                lines.append(f"{name}_count{_format_labels(labels)} {_number(buckets[-1])}")
        return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["Metrics", "metric_path"]
