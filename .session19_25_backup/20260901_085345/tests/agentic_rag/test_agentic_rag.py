import pytest
from src.agentic_rag.engine import answer_with_retrieval
from src.agentic_rag.models import Document
from src.agentic_rag.quality import evaluate_agentic_rag
pytestmark=pytest.mark.agentic_rag
DOCS=[Document('d1','The release process requires a deterministic quality gate.'),Document('d2','Agent retries distinguish decision failures from tool failures.'),Document('d3','The cafeteria opens at nine.')]
def test_rag_route():
    r=answer_with_retrieval('According to the documentation, what does the release process require?',DOCS,k=2); q=evaluate_agentic_rag(r,expected_retrieval=True,relevant_document_ids={'d1'}); assert q.recall_at_k==1 and q.release_passed
def test_no_retrieval(): assert evaluate_agentic_rag(answer_with_retrieval('Say hello',DOCS),expected_retrieval=False,relevant_document_ids=set()).release_passed
