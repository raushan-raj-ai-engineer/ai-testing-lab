from src.ai_api_testing.models import APIPerformanceReport
def percentile(values,p):
    if not values:return 0.0
    if not 0<=p<=1:raise ValueError('p must be between 0 and 1')
    o=sorted(values); i=(len(o)-1)*p; lo=int(i); hi=min(lo+1,len(o)-1); f=i-lo; return o[lo]*(1-f)+o[hi]*f
def build_api_performance_report(responses,*,min_success_rate=.99,max_p95_latency_ms=1000,max_latency_ms=3000):
    if not responses:raise ValueError('At least one API response is required')
    rate=sum(200<=r.status_code<300 for r in responses)/len(responses); l=[r.latency_ms for r in responses]; p50=percentile(l,.5); p95=percentile(l,.95); mx=max(l); f=[]
    if rate<min_success_rate:f.append('API success rate below threshold')
    if p95>max_p95_latency_ms:f.append('API p95 latency exceeded threshold')
    if mx>max_latency_ms:f.append('API max latency exceeded threshold')
    return APIPerformanceReport(len(responses),rate,p50,p95,mx,tuple(f),not f)
