import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.eval_pipeline.dashboard import write_html_dashboard
from src.eval_pipeline.pipeline import build_release_summary,load_eval_thresholds,load_records,write_release_report
def main():
    records=load_records(ROOT/'config/session21_sample_records.json'); th=load_eval_thresholds(ROOT/'config/session21_eval_thresholds.json'); s=build_release_summary(records,th); write_release_report(summary=s,records=records,path=ROOT/'reports/session21_evaluation_report.json'); write_html_dashboard(summary=s,records=records,path=ROOT/'reports/session21_dashboard.html'); print('SESSION 21 EVAL PIPELINE', 'PASS ✅' if s.release_passed else 'FAIL ❌'); raise SystemExit(0 if s.release_passed else 1)
if __name__=='__main__':main()
