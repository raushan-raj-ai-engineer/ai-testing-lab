import re
from src.agentic_rag.models import Document,RetrievalHit
def _tokens(text): return set(re.findall(r'[a-z0-9]+',text.casefold()))
def retrieve(query,documents,k=3):
    if k<1: raise ValueError('k must be >= 1')
    q=_tokens(query); out=[]
    for d in documents:
        tokens=_tokens(d.text); score=len(q&tokens)/max(len(q),1)
        if score>0: out.append(RetrievalHit(d.id,d.text,score))
    return sorted(out,key=lambda h:(-h.score,h.document_id))[:k]
