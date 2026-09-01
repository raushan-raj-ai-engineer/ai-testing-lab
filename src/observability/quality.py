from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from src.observability.models import ObservabilityBaseline, ObservabilityQualityReport, ObservabilityThresholds, ObservedSystemRun, TraceSpan

def load_observability_thresholds(path: str | Path) -> ObservabilityThresholds:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return ObservabilityThresholds(float(raw['max_total_latency_ms']), float(raw['max_browser_agent_latency_ms']),
        float(raw['max_single_mcp_call_latency_ms']), int(raw['max_failed_spans']), float(raw['max_mcp_failure_rate']),
        float(raw['max_retry_overhead_ratio']), bool(raw.get('require_usage_metrics', False)),
        None if raw.get('max_estimated_cost_usd') is None else float(raw['max_estimated_cost_usd']))

def load_observability_baseline(path: str | Path) -> ObservabilityBaseline:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    opt = lambda k: None if raw.get(k) is None else float(raw[k])
    return ObservabilityBaseline(opt('total_latency_ms'), opt('browser_agent_latency_ms'), opt('mcp_total_latency_ms'), float(raw.get('max_regression_ratio',1.5)))

def find_span(spans: tuple[TraceSpan,...], name: str) -> TraceSpan | None:
    matches = [s for s in spans if s.name == name]
    return max(matches, key=lambda s:s.duration_ms) if matches else None

def detect_latency_regressions(*, total_latency_ms: float, browser_agent_latency_ms: float, mcp_total_latency_ms: float,
                               baseline: ObservabilityBaseline | None) -> list[str]:
    if baseline is None: return []
    out=[]; ratio=baseline.max_regression_ratio
    for base,current,msg in ((baseline.total_latency_ms,total_latency_ms,'Total agent latency regressed'),
                             (baseline.browser_agent_latency_ms,browser_agent_latency_ms,'Browser-agent latency regressed'),
                             (baseline.mcp_total_latency_ms,mcp_total_latency_ms,'MCP latency regressed')):
        if base is not None and base>0 and current>base*ratio: out.append(msg)
    return out

def build_observability_report(run: ObservedSystemRun, thresholds: ObservabilityThresholds,
                               baseline: ObservabilityBaseline | None = None) -> ObservabilityQualityReport:
    spans=run.trace.spans; root=find_span(spans,'guarded_multi_agent_system'); browser=find_span(spans,'browser_agent.execute'); quality=find_span(spans,'quality_agent.execute')
    total=root.duration_ms if root else 0.0; browser_ms=browser.duration_ms if browser else 0.0; quality_ms=quality.duration_ms if quality else 0.0
    mcp=[s for s in spans if s.kind=='mcp_tool']; mcp_total=sum(s.duration_ms for s in mcp); mcp_count=len(mcp)
    mcp_fail=sum(s.status=='error' for s in mcp); mcp_rate=mcp_fail/mcp_count if mcp_count else 0.0
    slow=max(mcp,key=lambda s:s.duration_ms) if mcp else None; slow_name=str(slow.attributes.get('tool_name',slow.name)) if slow else None; slow_ms=slow.duration_ms if slow else 0.0
    failed_spans=sum(s.status=='error' for s in spans)
    retry_total=run.decision_attempts+run.execution_attempts; retry_bad=run.rejected_decisions+run.execution_failures; retry_ratio=retry_bad/retry_total if retry_total else 0.0
    usage=run.trace.usage; usage_available=bool(usage); input_tokens=output_tokens=None; cost=None
    if usage:
        ins=[u.input_tokens for u in usage if u.input_tokens is not None]; outs=[u.output_tokens for u in usage if u.output_tokens is not None]; costs=[u.estimated_cost_usd for u in usage if u.estimated_cost_usd is not None]
        if len(ins)==len(usage): input_tokens=sum(ins)
        if len(outs)==len(usage): output_tokens=sum(outs)
        if len(costs)==len(usage): cost=sum(costs)
    trace_complete=all((root is not None,browser is not None,quality is not None))
    regressions=detect_latency_regressions(total_latency_ms=total,browser_agent_latency_ms=browser_ms,mcp_total_latency_ms=mcp_total,baseline=baseline)
    failures=[]
    if not run.functional_release_passed: failures.append('Functional/safety release gate failed')
    if not trace_complete: failures.append('Required observability spans are missing')
    if total>thresholds.max_total_latency_ms: failures.append('Total latency exceeded budget')
    if browser_ms>thresholds.max_browser_agent_latency_ms: failures.append('Browser-agent latency exceeded budget')
    if slow_ms>thresholds.max_single_mcp_call_latency_ms: failures.append('Single MCP call exceeded latency budget')
    if failed_spans>thresholds.max_failed_spans: failures.append('Failed span count exceeded threshold')
    if mcp_rate>thresholds.max_mcp_failure_rate: failures.append('MCP failure rate exceeded threshold')
    if retry_ratio>thresholds.max_retry_overhead_ratio: failures.append('Retry overhead exceeded threshold')
    if thresholds.require_usage_metrics and not usage_available: failures.append('Token/cost usage metrics are required but unavailable')
    if thresholds.max_estimated_cost_usd is not None and cost is not None and cost>thresholds.max_estimated_cost_usd: failures.append('Estimated LLM cost exceeded budget')
    failures.extend(regressions)
    return ObservabilityQualityReport(run.trace.trace_id,trace_complete,run.functional_release_passed,total,browser_ms,quality_ms,mcp_total,mcp_count,mcp_fail,mcp_rate,slow_name,slow_ms,failed_spans,run.decision_attempts,run.rejected_decisions,run.execution_attempts,run.execution_failures,retry_ratio,usage_available,input_tokens,output_tokens,cost,tuple(regressions),tuple(failures),not failures)

def write_observability_report(*, run: ObservedSystemRun, report: ObservabilityQualityReport, path: str | Path) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'report':asdict(report),'trace':{'trace_id':run.trace.trace_id,'spans':[asdict(s) for s in run.trace.spans],'usage':[asdict(u) for u in run.trace.usage]}},indent=2),encoding='utf-8')

def write_observability_baseline(*, report: ObservabilityQualityReport, path: str | Path, max_regression_ratio: float=1.5) -> None:
    Path(path).write_text(json.dumps({'total_latency_ms':report.total_latency_ms,'browser_agent_latency_ms':report.browser_agent_latency_ms,'mcp_total_latency_ms':report.mcp_total_latency_ms,'max_regression_ratio':max_regression_ratio},indent=2),encoding='utf-8')
