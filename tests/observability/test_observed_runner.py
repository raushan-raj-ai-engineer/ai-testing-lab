import pytest
from src.browser_agent.browser_agent import BrowserAgentRunResult
from src.observability.runner import run_observed_guarded_system
pytestmark=[pytest.mark.observability,pytest.mark.asyncio]
class FakeClient: pass
class FakeBrowser:
    async def execute(self,client,task):
        todo=task.payload['todo_name']; return BrowserAgentRunResult(True,[],f'''- checkbox "Toggle Todo" [checked] [ref=e21]
- generic [ref=e22]: {todo}''',[],[])
async def test_runner_spans():
    r=await run_observed_guarded_system(FakeClient(),'Add Buy milk and mark it complete',browser_executor=FakeBrowser()); names={s.name for s in r.trace.spans}; assert {'guarded_multi_agent_system','browser_agent.execute','quality_agent.execute'}<=names
