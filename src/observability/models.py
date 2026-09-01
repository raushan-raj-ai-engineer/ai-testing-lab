from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from src.guardrails.models import GuardedMultiAgentRunResult

@dataclass(frozen=True)
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: str
    ended_at: str
    duration_ms: float
    status: str
    error: str | None
    attributes: dict[str, Any]

@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    operation: str | None = None

@dataclass(frozen=True)
class TraceSnapshot:
    trace_id: str
    spans: tuple[TraceSpan, ...]
    usage: tuple[UsageRecord, ...]

@dataclass(frozen=True)
class ObservedSystemRun:
    trace: TraceSnapshot
    guarded_result: GuardedMultiAgentRunResult | None
    functional_release_passed: bool
    decision_attempts: int
    rejected_decisions: int
    execution_attempts: int
    execution_failures: int

@dataclass(frozen=True)
class ObservabilityThresholds:
    max_total_latency_ms: float
    max_browser_agent_latency_ms: float
    max_single_mcp_call_latency_ms: float
    max_failed_spans: int
    max_mcp_failure_rate: float
    max_retry_overhead_ratio: float
    require_usage_metrics: bool = False
    max_estimated_cost_usd: float | None = None

@dataclass(frozen=True)
class ObservabilityBaseline:
    total_latency_ms: float | None
    browser_agent_latency_ms: float | None
    mcp_total_latency_ms: float | None
    max_regression_ratio: float

@dataclass(frozen=True)
class ObservabilityQualityReport:
    trace_id: str
    trace_complete: bool
    functional_release_passed: bool
    total_latency_ms: float
    browser_agent_latency_ms: float
    quality_agent_latency_ms: float
    mcp_total_latency_ms: float
    mcp_call_count: int
    mcp_failure_count: int
    mcp_failure_rate: float
    slowest_mcp_call_name: str | None
    slowest_mcp_call_latency_ms: float
    failed_span_count: int
    decision_attempts: int
    rejected_decisions: int
    execution_attempts: int
    execution_failures: int
    retry_overhead_ratio: float
    usage_metrics_available: bool
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    regressions: tuple[str, ...]
    release_failures: tuple[str, ...]
    release_passed: bool
