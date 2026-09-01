from src.agentic_rag.models import AgenticRAGResult
from src.agentic_rag.retriever import retrieve
def should_retrieve(query):
    q=query.casefold(); return any(x in q for x in ('according to','policy','documentation','manual','knowledge base','release process'))
def answer_with_retrieval(query,documents,*,k=3):
    if not should_retrieve(query): return AgenticRAGResult(query,False,(),'No retrieval required for this query.',())
    hits=tuple(retrieve(query,documents,k));
    if not hits: return AgenticRAGResult(query,True,(),'Insufficient retrieved evidence.',())
    return AgenticRAGResult(query,True,hits,' '.join(h.text for h in hits),tuple(h.document_id for h in hits))
