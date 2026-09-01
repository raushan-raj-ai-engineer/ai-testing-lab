import pytest
from src.observability.models import ObservabilityBaseline,ObservabilityThresholds,ObservedSystemRun,TraceSnapshot,TraceSpan
from src.observability.quality import build_observability_report
pytestmark=pytest.mark.observability

def s(name,kind,duration,tool=None):
    return TraceSpan('t','s-'+name,None,name,kind,'a','b',duration,'ok',None,{} if tool is None else {'tool_name':tool})
def run(): return ObservedSystemRun(TraceSnapshot('t',(s('guarded_multi_agent_system','orchestrator',1000),s('browser_agent.execute','agent',800),s('quality_agent.execute','agent',50),s('mcp.browser_type','mcp_tool',100,'browser_type')),()),None,True,2,0,2,0)
def th(): return ObservabilityThresholds(5000,4000,1000,1,.5,.5)
def test_pass(): assert build_observability_report(run(),th()).release_passed is True
def test_regression():
    b=ObservabilityBaseline(500,400,50,1.5); r=build_observability_report(run(),th(),b); assert r.release_passed is False
def test_no_fake_usage(): assert build_observability_report(run(),th()).input_tokens is None
