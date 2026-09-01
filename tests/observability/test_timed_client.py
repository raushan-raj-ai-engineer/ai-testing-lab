from dataclasses import dataclass
import pytest
from src.observability.timed_client import TimedMCPClient
from src.observability.tracer import TraceRecorder
pytestmark=[pytest.mark.observability,pytest.mark.asyncio]
@dataclass
class R: is_error: bool=False
class C:
    async def call_tool(self,name,arguments): return R(False)
    async def list_tools(self): return []
class E:
    async def call_tool(self,name,arguments): return R(True)
async def test_timed_call():
    t=TraceRecorder(); await TimedMCPClient(C(),t).call_tool('browser_click',{'target':'e1'}); assert t.snapshot().spans[0].kind=='mcp_tool'
async def test_error_marked():
    t=TraceRecorder(); await TimedMCPClient(E(),t).call_tool('browser_click',{}); assert t.snapshot().spans[0].status=='error'
