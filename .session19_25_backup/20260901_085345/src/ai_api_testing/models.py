from __future__ import annotations
from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class APIResponse: status_code:int; json_body:dict[str,Any]; latency_ms:float
@dataclass(frozen=True)
class ContractIssue: path:str; message:str
@dataclass(frozen=True)
class APIPerformanceReport: total_requests:int; success_rate:float; p50_latency_ms:float; p95_latency_ms:float; max_latency_ms:float; release_failures:tuple[str,...]; release_passed:bool
