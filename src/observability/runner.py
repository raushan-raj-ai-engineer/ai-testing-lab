from __future__ import annotations
from typing import Any
from src.guardrails.quality import build_runtime_guardrail_report
from src.guardrails.safe_orchestrator import run_guarded_multi_agent_system
from src.observability.executors import ObservedBrowserExecutor, ObservedQualityExecutor
from src.observability.models import ObservedSystemRun
from src.observability.timed_client import TimedMCPClient
from src.observability.tracer import TraceRecorder

async def run_observed_guarded_system(client: Any, goal: str, tracer: TraceRecorder | None = None,
                                      browser_executor: Any | None = None, quality_executor: Any | None = None) -> ObservedSystemRun:
    tracer = tracer or TraceRecorder(); timed = TimedMCPClient(client, tracer)
    browser = ObservedBrowserExecutor(tracer, browser_executor); quality = ObservedQualityExecutor(tracer, quality_executor)
    with tracer.span('guarded_multi_agent_system', 'orchestrator', {'goal_length':len(goal)}) as root:
        guarded = await run_guarded_multi_agent_system(client=timed, goal=goal, browser_executor=browser, quality_executor=quality)
        root.set_attribute('completed', guarded.completed)
    runtime = build_runtime_guardrail_report(guarded)
    da = rd = ea = ef = 0
    multi = guarded.multi_agent_result
    if multi is not None and multi.browser_result is not None:
        br = multi.browser_result
        da = len(br.attempts); rd = sum(not a.accepted for a in br.attempts)
        ea = len(br.execution_attempts); ef = sum(not a.succeeded for a in br.execution_attempts)
    return ObservedSystemRun(tracer.snapshot(), guarded, runtime.safety_release_passed, da, rd, ea, ef)
