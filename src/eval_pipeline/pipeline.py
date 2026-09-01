from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from src.eval_pipeline.models import EvalThresholds,EvaluationRecord,ReleaseSummary

def load_eval_thresholds(path):
    r=json.loads(Path(path).read_text(encoding='utf-8')); return EvalThresholds(float(r['min_pass_rate']),float(r['min_average_score']),int(r['max_failed_scenarios']),int(r['max_regressions']))
def load_records(path):
    r=json.loads(Path(path).read_text(encoding='utf-8')); items=r['records'] if isinstance(r,dict) else r; return [EvaluationRecord(i['scenario_id'],i['category'],bool(i['passed']),float(i['score']),None if i.get('latency_ms') is None else float(i['latency_ms']),i.get('notes')) for i in items]
def detect_record_regressions(current,baseline,score_drop_tolerance=.10):
    bm={r.scenario_id:r for r in baseline}; out=[]
    for r in current:
        p=bm.get(r.scenario_id)
        if not p: continue
        if p.passed and not r.passed: out.append(f'{r.scenario_id}: pass -> fail')
        if r.score<p.score-score_drop_tolerance: out.append(f'{r.scenario_id}: score regressed')
    return out
def build_release_summary(records,thresholds,regressions=None):
    if not records: raise ValueError('At least one evaluation record is required')
    regressions=list(regressions or []); total=len(records); passed=sum(r.passed for r in records); failed=total-passed; rate=passed/total; avg=mean(r.score for r in records); cats={}
    for c in sorted({r.category for r in records}):
        x=[r for r in records if r.category==c]; cats[c]=sum(r.passed for r in x)/len(x)
    f=[]
    if rate<thresholds.min_pass_rate:f.append('Evaluation pass rate below threshold')
    if avg<thresholds.min_average_score:f.append('Average evaluation score below threshold')
    if failed>thresholds.max_failed_scenarios:f.append('Failed scenario count exceeded threshold')
    if len(regressions)>thresholds.max_regressions:f.append('Regression count exceeded threshold')
    return ReleaseSummary(total,passed,failed,rate,avg,cats,tuple(regressions),tuple(f),not f)
def write_release_report(*,summary,records,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'summary':asdict(summary),'records':[asdict(r) for r in records]},indent=2),encoding='utf-8')
