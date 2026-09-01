from pathlib import Path
from uuid import uuid4
import pytest
from mcp import Client
from src.observability.quality import build_observability_report,load_observability_baseline,load_observability_thresholds
from src.observability.runner import run_observed_guarded_system
pytestmark=[pytest.mark.observability_live,pytest.mark.asyncio]
ROOT=Path(__file__).resolve().parents[2]
async def test_real_observability_gate():
    goal=f'Add Session19-{uuid4().hex[:6]} and mark it complete'; th=load_observability_thresholds(ROOT/'config/session19_observability_thresholds.json'); b=load_observability_baseline(ROOT/'config/session19_observability_baseline.json')
    async with Client('http://localhost:8931/mcp') as c: run=await run_observed_guarded_system(c,goal)
    r=build_observability_report(run,th,b); assert r.trace_complete and r.functional_release_passed and r.release_passed
