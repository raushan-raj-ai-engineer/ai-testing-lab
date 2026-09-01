from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Document: id:str; text:str
@dataclass(frozen=True)
class RetrievalHit: document_id:str; text:str; score:float
@dataclass(frozen=True)
class AgenticRAGResult: query:str; retrieval_used:bool; hits:tuple[RetrievalHit,...]; answer:str; citations:tuple[str,...]
@dataclass(frozen=True)
class AgenticRAGQualityReport: route_correct:bool; precision_at_k:float; recall_at_k:float; citation_precision:float; citation_recall:float; grounding_score:float; release_passed:bool
