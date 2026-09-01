from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class ReliabilityRun:
    run_id:str; passed:bool; latency_ms:float; decision_attempts:int=0; rejected_decisions:int=0; execution_attempts:int=0; execution_failures:int=0; error_type:str|None=None
@dataclass(frozen=True)
class ReliabilityThresholds:
    min_pass_rate:float; max_flake_rate:float; max_latency_cv:float; max_average_retry_overhead:float; max_consecutive_failures:int
@dataclass(frozen=True)
class ReliabilityReport:
    total_runs:int; passed_runs:int; failed_runs:int; pass_rate:float; flake_rate:float; mean_latency_ms:float; latency_stddev_ms:float; latency_cv:float; average_retry_overhead:float; max_consecutive_failures:int; failure_types:tuple[str,...]; release_failures:tuple[str,...]; release_passed:bool
