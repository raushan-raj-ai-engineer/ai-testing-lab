from statistics import mean
from src.final_project.models import FinalReleaseReport
def build_final_release_report(components,*,min_average_score=.8,max_noncritical_failures=1):
    if not components:raise ValueError('At least one component gate is required')
    passed=sum(c.passed for c in components); failed=len(components)-passed; avg=mean(c.score for c in components); critical=tuple(c.name for c in components if c.critical and not c.passed); non=[c for c in components if not c.critical and not c.passed]; f=[]
    if critical:f.append('One or more critical AI quality gates failed')
    if avg<min_average_score:f.append('Average system quality score below threshold')
    if len(non)>max_noncritical_failures:f.append('Too many non-critical component failures')
    return FinalReleaseReport(len(components),passed,failed,avg,critical,tuple(f),not f)
