import json,random
from pathlib import Path
from src.interview_prep.models import AnswerScore,InterviewQuestion
def load_question_bank(path):
    r=json.loads(Path(path).read_text(encoding='utf-8')); return [InterviewQuestion(i['id'],i['topic'],i['difficulty'],i['question'],tuple(i['key_points']),i['model_answer']) for i in r['questions']]
def select_questions(questions,*,topic=None,difficulty=None,count=5,seed=42):
    q=[x for x in questions if (topic is None or x.topic==topic) and (difficulty is None or x.difficulty==difficulty)]; rng=random.Random(seed); rng.shuffle(q); return q[:count]
def score_answer(question,answer):
    a=answer.casefold(); matched=tuple(p for p in question.key_points if p.casefold() in a); missing=tuple(p for p in question.key_points if p not in matched); return AnswerScore(question.id,matched,missing,len(matched)/len(question.key_points) if question.key_points else 1.0)
