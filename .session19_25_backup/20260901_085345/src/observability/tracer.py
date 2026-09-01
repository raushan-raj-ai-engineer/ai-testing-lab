from __future__ import annotations
import contextvars, threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Any, Iterator
from uuid import uuid4
from src.observability.models import TraceSnapshot, TraceSpan, UsageRecord

_CURRENT_SPAN: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar('observability_current_span', default=None)

@dataclass
class SpanHandle:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = 'ok'
    error: str | None = None
    def set_attribute(self, key: str, value: Any) -> None: self.attributes[key] = value
    def mark_error(self, error: str) -> None:
        self.status = 'error'; self.error = error

class TraceRecorder:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or f'trace-{uuid4().hex}'
        self._spans: list[TraceSpan] = []
        self._usage: list[UsageRecord] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, kind: str, attributes: dict[str, Any] | None = None) -> Iterator[SpanHandle]:
        current = _CURRENT_SPAN.get()
        parent = current[1] if current is not None and current[0] == self.trace_id else None
        span_id = f'span-{uuid4().hex[:12]}'
        handle = SpanHandle(self.trace_id, span_id, parent, name, kind, dict(attributes or {}))
        started_at = datetime.now(timezone.utc).isoformat(); started_ns = perf_counter_ns()
        token = _CURRENT_SPAN.set((self.trace_id, span_id))
        try:
            yield handle
        except BaseException as exc:
            handle.mark_error(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            ended_ns = perf_counter_ns(); ended_at = datetime.now(timezone.utc).isoformat()
            completed = TraceSpan(self.trace_id, span_id, parent, name, kind, started_at, ended_at,
                                  (ended_ns-started_ns)/1_000_000, handle.status, handle.error, dict(handle.attributes))
            with self._lock: self._spans.append(completed)
            _CURRENT_SPAN.reset(token)

    def record_usage(self, *, provider: str, model: str, input_tokens: int | None,
                     output_tokens: int | None, estimated_cost_usd: float | None,
                     operation: str | None = None) -> None:
        with self._lock:
            self._usage.append(UsageRecord(provider, model, input_tokens, output_tokens, estimated_cost_usd, operation))

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            return TraceSnapshot(self.trace_id, tuple(self._spans), tuple(self._usage))
