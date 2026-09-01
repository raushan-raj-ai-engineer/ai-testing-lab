from pathlib import Path
import pytest
from src.interview_prep.engine import load_question_bank,score_answer,select_questions
pytestmark=pytest.mark.interview_prep
ROOT=Path(__file__).resolve().parents[2]
def test_load(): assert len(load_question_bank(ROOT/'config/session25_interview_questions.json'))>=20
def test_filter():
    q=load_question_bank(ROOT/'config/session25_interview_questions.json'); s=select_questions(q,topic='agent',count=10); assert s and all(x.topic=='agent' for x in s)
def test_score():
    q=next(x for x in load_question_bank(ROOT/'config/session25_interview_questions.json') if x.id=='q04'); r=score_answer(q,'Semantic invalidity needs re-planning; transient failure can retry the same action.'); assert r.score>=.75
