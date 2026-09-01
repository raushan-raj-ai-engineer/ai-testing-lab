from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.final_project.models import ComponentGate
from src.final_project.release_gate import build_final_release_report

def read(path: Path): return json.loads(path.read_text(encoding='utf-8'))
def gate(name,path,pass_path,score_path=None,critical=True):
    if not path.exists(): return ComponentGate(name,False,0.0,critical)
    raw=read(path); value=raw
    for k in pass_path:value=value[k]
    score=1.0
    if score_path:
        score=raw
        for k in score_path:score=score[k]
        score=float(score)
    return ComponentGate(name,bool(value),score,critical)

def main():
    components=[
        gate('safety',ROOT/'reports/session18_guardrail_report.json',('summary','release_passed'),('summary','detection_rate'),True),
        gate('observability',ROOT/'reports/session19_observability_report.json',('report','release_passed'),None,True),
        gate('reliability',ROOT/'reports/session20_reliability_report.json',('report','release_passed'),('report','pass_rate'),True),
        gate('evaluation_pipeline',ROOT/'reports/session21_evaluation_report.json',('summary','release_passed'),('summary','average_score'),True),
        gate('agentic_rag',ROOT/'reports/session22_agentic_rag_report.json',('release_passed',),('grounding_score',),True),
        gate('ai_api',ROOT/'reports/session23_api_quality_report.json',('release_passed',),('performance','success_rate'),True),
    ]
    r=build_final_release_report(components)
    print('SESSION 24 FINAL AI SDET RELEASE GATE')
    for c in components: print('-',c.name, 'PASS' if c.passed else 'FAIL', f'score={c.score:.2f}')
    print('Critical failures:',r.critical_failures); print('Average score:',round(r.average_score,3)); print('FINAL RELEASE:','PASS ✅' if r.release_passed else 'FAIL ❌')
    if not r.release_passed: print('Tip: run Sessions 18-23 report-producing scripts before Session 24.')
    raise SystemExit(0 if r.release_passed else 1)
if __name__=='__main__':main()
