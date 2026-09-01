from __future__ import annotations
from typing import Any
from src.browser_agent.advanced_quality import AdvancedAgentQualityReport
from src.browser_agent.browser_agent import BrowserAgentRunResult
from src.multi_agent.executors import BrowserAgentExecutor, QualityAgentExecutor
from src.multi_agent.models import AgentTask
from src.observability.tracer import TraceRecorder

class ObservedBrowserExecutor:
    def __init__(self, tracer: TraceRecorder, inner: Any | None = None) -> None:
        self.tracer = tracer; self.inner = inner or BrowserAgentExecutor()
    async def execute(self, client: Any, task: AgentTask) -> BrowserAgentRunResult:
        with self.tracer.span('browser_agent.execute', 'agent', {'agent':'browser_agent','task_id':task.id}) as span:
            result = await self.inner.execute(client=client, task=task)
            span.set_attribute('completed', result.completed)
            span.set_attribute('successful_steps', len(result.steps))
            span.set_attribute('decision_attempts', len(result.attempts))
            span.set_attribute('execution_attempts', len(result.execution_attempts))
            return result

class ObservedQualityExecutor:
    def __init__(self, tracer: TraceRecorder, inner: Any | None = None) -> None:
        self.tracer = tracer; self.inner = inner or QualityAgentExecutor()
    async def execute(self, task: AgentTask, browser_result: BrowserAgentRunResult) -> AdvancedAgentQualityReport:
        with self.tracer.span('quality_agent.execute', 'agent', {'agent':'quality_agent','task_id':task.id}) as span:
            result = await self.inner.execute(task=task, browser_result=browser_result)
            span.set_attribute('completed', result.completed)
            span.set_attribute('decision_efficiency', result.decision_efficiency)
            return result
