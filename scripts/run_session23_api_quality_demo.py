import json,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.ai_api_testing.contract import validate_contract
from src.ai_api_testing.models import APIResponse
from src.ai_api_testing.performance import build_api_performance_report
def main():
    schema={'required':['answer','confidence'],'additionalProperties':False,'properties':{'answer':{'type':'string'},'confidence':{'type':'number'}}}; payload={'answer':'release passed','confidence':.97}; issues=validate_contract(payload,schema); perf=build_api_performance_report([APIResponse(200,payload,x) for x in (80,90,100,110,120)],min_success_rate=1,max_p95_latency_ms=200,max_latency_ms=250); passed=not issues and perf.release_passed
    out=ROOT/'reports/session23_api_quality_report.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'contract_issues':[asdict(x) for x in issues],'performance':asdict(perf),'release_passed':passed},indent=2),encoding='utf-8')
    print('SESSION 23 AI API QUALITY'); print('Contract issues:',issues); print('P95:',round(perf.p95_latency_ms,2)); print('Report:',out); print('Release:','PASS ✅' if passed else 'FAIL ❌'); raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
