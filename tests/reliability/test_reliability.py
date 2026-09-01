import pytest
from src.reliability.models import ReliabilityRun,ReliabilityThresholds
from src.reliability.quality import build_reliability_report
pytestmark=pytest.mark.reliability
def th(): return ReliabilityThresholds(.8,.2,.5,.5,1)
def test_stable_pass(): assert build_reliability_report([ReliabilityRun(str(i),True,100+i,2,0,2,0) for i in range(5)],th()).release_passed
def test_flaky_fail():
    runs=[ReliabilityRun('1',True,100),ReliabilityRun('2',False,100,error_type='RuntimeError'),ReliabilityRun('3',True,100),ReliabilityRun('4',False,100,error_type='RuntimeError'),ReliabilityRun('5',True,100)]
    r=build_reliability_report(runs,th()); assert r.flake_rate==.4 and not r.release_passed
def test_retry_fail(): assert not build_reliability_report([ReliabilityRun('1',True,100,4,3,2,1)],th()).release_passed
