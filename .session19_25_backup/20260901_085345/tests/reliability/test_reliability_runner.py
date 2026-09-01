import pytest
from src.reliability.runner import run_repeated_async
pytestmark=[pytest.mark.reliability,pytest.mark.asyncio]
async def test_runner():
    async def op(): return {'ok':True}
    r=await run_repeated_async(op,runs=3,success=lambda v:v['ok']); assert len(r)==3 and all(x.passed for x in r)
async def test_exception_capture():
    async def op(): raise RuntimeError('boom')
    r=await run_repeated_async(op,runs=2,success=lambda _:True); assert all(x.error_type=='RuntimeError' for x in r)
