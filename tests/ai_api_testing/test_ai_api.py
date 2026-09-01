import pytest
from src.ai_api_testing.contract import validate_contract
from src.ai_api_testing.models import APIResponse
from src.ai_api_testing.performance import build_api_performance_report
pytestmark=pytest.mark.ai_api
SCHEMA={'required':['answer','confidence'],'additionalProperties':False,'properties':{'answer':{'type':'string'},'confidence':{'type':'number'}}}
def test_contract(): assert validate_contract({'answer':'ok','confidence':.9},SCHEMA)==[]
def test_contract_failure(): assert len(validate_contract({'answer':'ok','extra':1},SCHEMA))==2
def test_perf(): assert build_api_performance_report([APIResponse(200,{},x) for x in (100,110,120)],min_success_rate=1,max_p95_latency_ms=200,max_latency_ms=250).release_passed
