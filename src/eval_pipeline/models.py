from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class EvaluationRecord: scenario_id:str; category:str; passed:bool; score:float; latency_ms:float|None=None; notes:str|None=None
@dataclass(frozen=True)
class EvalThresholds: min_pass_rate:float; min_average_score:float; max_failed_scenarios:int; max_regressions:int
@dataclass(frozen=True)
class ReleaseSummary: total:int; passed:int; failed:int; pass_rate:float; average_score:float; category_pass_rates:dict[str,float]; regressions:tuple[str,...]; release_failures:tuple[str,...]; release_passed:bool
