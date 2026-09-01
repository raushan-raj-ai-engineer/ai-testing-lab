from __future__ import annotations
import argparse,asyncio,sys
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from mcp import Client
from src.observability.quality import build_observability_report,load_observability_baseline,load_observability_thresholds,write_observability_baseline,write_observability_report
from src.observability.runner import run_observed_guarded_system
async def go(args):
    th=load_observability_thresholds(ROOT/'config/session19_observability_thresholds.json'); b=load_observability_baseline(ROOT/'config/session19_observability_baseline.json'); goal=f'Add Session19-{uuid4().hex[:6]} and mark it complete'
    async with Client(args.mcp_url) as c: run=await run_observed_guarded_system(c,goal)
    r=build_observability_report(run,th,b); write_observability_report(run=run,report=r,path=args.report)
    print('SESSION 19 OBSERVABILITY'); print('Trace:',r.trace_id); print('Total latency ms:',round(r.total_latency_ms,2)); print('MCP calls:',r.mcp_call_count); print('Retry overhead:',f'{r.retry_overhead_ratio:.2%}'); print('Usage available:',r.usage_metrics_available); print('Release:','PASS ✅' if r.release_passed else 'FAIL ❌')
    if args.update_baseline and r.release_passed: write_observability_baseline(report=r,path=ROOT/'config/session19_observability_baseline.json')
    return 0 if r.release_passed else 1
def main():
    p=argparse.ArgumentParser(); p.add_argument('--mcp-url',default='http://localhost:8931/mcp'); p.add_argument('--report',default=str(ROOT/'reports/session19_observability_report.json')); p.add_argument('--update-baseline',action='store_true'); a=p.parse_args(); raise SystemExit(asyncio.run(go(a)))
if __name__=='__main__': main()
