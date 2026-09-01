from __future__ import annotations
import json
from pathlib import Path
from statistics import mean,pstdev
from src.reliability.models import ReliabilityReport,ReliabilityRun,ReliabilityThresholds

def load_reliability_thresholds(path:str|Path)->ReliabilityThresholds:
    r=json.loads(Path(path).read_text(encoding='utf-8')); return ReliabilityThresholds(float(r['min_pass_rate']),float(r['max_flake_rate']),float(r['max_latency_cv']),float(r['max_average_retry_overhead']),int(r['max_consecutive_failures']))
def _streak(runs):
    cur=best=0
    for r in runs:
        if r.passed: cur=0
        else: cur+=1; best=max(best,cur)
    return best
def _retry(r):
    total=r.decision_attempts+r.execution_attempts; bad=r.rejected_decisions+r.execution_failures; return bad/total if total else 0.0
def build_reliability_report(runs:list[ReliabilityRun],thresholds:ReliabilityThresholds)->ReliabilityReport:
    if not runs: raise ValueError('At least one reliability run is required')
    total=len(runs); passed=sum(r.passed for r in runs); failed=total-passed; pass_rate=passed/total; flake=0.0 if passed in (0,total) else failed/total
    l=[r.latency_ms for r in runs]; avg=mean(l); sd=pstdev(l); cv=sd/avg if avg>0 else 0.0; retry=mean(_retry(r) for r in runs); streak=_streak(runs); types=tuple(sorted({r.error_type for r in runs if r.error_type}))
    failures=[]
    if pass_rate<thresholds.min_pass_rate: failures.append('Reliability pass rate below threshold')
    if flake>thresholds.max_flake_rate: failures.append('Flake rate exceeded threshold')
    if cv>thresholds.max_latency_cv: failures.append('Latency variability exceeded threshold')
    if retry>thresholds.max_average_retry_overhead: failures.append('Average retry overhead exceeded threshold')
    if streak>thresholds.max_consecutive_failures: failures.append('Consecutive failure streak exceeded threshold')
    return ReliabilityReport(total,passed,failed,pass_rate,flake,avg,sd,cv,retry,streak,types,tuple(failures),not failures)
