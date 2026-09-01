import pytest
from src.observability.tracer import TraceRecorder
pytestmark=pytest.mark.observability

def test_nested_spans():
    t=TraceRecorder('trace-test')
    with t.span('root','orchestrator'):
        with t.span('child','agent'): pass
    spans=t.snapshot().spans; root=next(s for s in spans if s.name=='root'); child=next(s for s in spans if s.name=='child')
    assert child.parent_span_id==root.span_id

def test_exception_span():
    t=TraceRecorder()
    with pytest.raises(RuntimeError):
        with t.span('broken','agent'): raise RuntimeError('boom')
    assert t.snapshot().spans[0].status=='error'

def test_usage_optional():
    t=TraceRecorder(); t.record_usage(provider='x',model='m',input_tokens=10,output_tokens=2,estimated_cost_usd=.01)
    assert t.snapshot().usage[0].input_tokens==10
