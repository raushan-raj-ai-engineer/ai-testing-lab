from __future__ import annotations
import argparse,asyncio,json,sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from mcp import Client
from src.observability.quality import build_observability_report,load_observability_baseline,load_observability_thresholds
from src.observability.runner import run_observed_guarded_system
from src.reliability.quality import build_reliability_report,load_reliability_thresholds
from src.reliability.runner import run_repeated_async
async def go(a):
    ot=load_observability_thresholds(ROOT/'config/session19_observability_thresholds.json'); b=load_observability_baseline(ROOT/'config/session19_observability_baseline.json'); rt=load_reliability_thresholds(ROOT/'config/session20_reliability_thresholds.json')
    async with Client(a.mcp_url) as c:
        async def op():
            run=await run_observed_guarded_system(c,f'Add Session20-{uuid4().hex[:6]} and mark it complete'); return run,build_observability_report(run,ot,b)
        runs=await run_repeated_async(op,runs=a.runs,success=lambda v:v[1].release_passed,telemetry=lambda v:{'decision_attempts':v[0].decision_attempts,'rejected_decisions':v[0].rejected_decisions,'execution_attempts':v[0].execution_attempts,'execution_failures':v[0].execution_failures})
    r=build_reliability_report(runs,rt)
    out=ROOT/'reports/session20_reliability_report.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'report':asdict(r),'runs':[asdict(x) for x in runs]},indent=2),encoding='utf-8')
    print('SESSION 20 RELIABILITY'); print('Pass rate:',f'{r.pass_rate:.2%}'); print('Flake rate:',f'{r.flake_rate:.2%}'); print('Latency CV:',round(r.latency_cv,3)); print('Report:',out); print('Release:','PASS ✅' if r.release_passed else 'FAIL ❌'); return 0 if r.release_passed else 1
def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',type=int,default=3); p.add_argument('--mcp-url',default='http://localhost:8931/mcp'); a=p.parse_args(); raise SystemExit(asyncio.run(go(a)))
if __name__=='__main__': main()
