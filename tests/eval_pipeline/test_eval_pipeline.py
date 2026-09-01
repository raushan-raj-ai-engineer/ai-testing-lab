import pytest
from src.eval_pipeline.models import EvalThresholds,EvaluationRecord
from src.eval_pipeline.pipeline import build_release_summary,detect_record_regressions
pytestmark=pytest.mark.eval_pipeline
def test_summary(): assert build_release_summary([EvaluationRecord('a','agent',True,1),EvaluationRecord('b','rag',True,.9)],EvalThresholds(.8,.8,1,0)).release_passed
def test_regression():
    r=detect_record_regressions([EvaluationRecord('a','agent',False,.5)],[EvaluationRecord('a','agent',True,1)]); assert len(r)==2
