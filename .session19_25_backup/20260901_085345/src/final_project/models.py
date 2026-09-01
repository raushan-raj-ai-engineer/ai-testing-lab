from dataclasses import dataclass
@dataclass(frozen=True)
class ComponentGate: name:str; passed:bool; score:float; critical:bool=True
@dataclass(frozen=True)
class FinalReleaseReport: total_components:int; passed_components:int; failed_components:int; average_score:float; critical_failures:tuple[str,...]; release_failures:tuple[str,...]; release_passed:bool
