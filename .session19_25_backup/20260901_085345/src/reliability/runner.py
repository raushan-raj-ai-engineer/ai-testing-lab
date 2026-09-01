from __future__ import annotations
from time import perf_counter_ns
from typing import Awaitable,Callable,TypeVar
from uuid import uuid4
from src.reliability.models import ReliabilityRun
T=TypeVar('T')
async def run_repeated_async(operation:Callable[[],Awaitable[T]],*,runs:int,success:Callable[[T],bool],telemetry:Callable[[T],dict[str,int]]|None=None)->list[ReliabilityRun]:
    if runs<1: raise ValueError('runs must be >= 1')
    out=[]
    for _ in range(runs):
        start=perf_counter_ns()
        try:
            value=await operation(); passed=bool(success(value)); data=telemetry(value) if telemetry else {}; error=None if passed else 'quality_gate_failure'
        except Exception as exc:
            passed=False; data={}; error=type(exc).__name__
        out.append(ReliabilityRun(f'run-{uuid4().hex[:8]}',passed,(perf_counter_ns()-start)/1_000_000,int(data.get('decision_attempts',0)),int(data.get('rejected_decisions',0)),int(data.get('execution_attempts',0)),int(data.get('execution_failures',0)),error))
    return out
