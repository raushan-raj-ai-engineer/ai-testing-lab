import pytest
from src.final_project.models import ComponentGate
from src.final_project.release_gate import build_final_release_report
pytestmark=pytest.mark.final_project
def test_pass(): assert build_final_release_report([ComponentGate('functional',True,1),ComponentGate('safety',True,1),ComponentGate('rag',True,.9)]).release_passed
def test_critical_fail(): assert not build_final_release_report([ComponentGate('functional',True,1),ComponentGate('safety',False,0,True)]).release_passed
