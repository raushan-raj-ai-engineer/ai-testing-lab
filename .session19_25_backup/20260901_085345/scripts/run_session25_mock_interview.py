import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.interview_prep.engine import load_question_bank,select_questions
def main():
    p=argparse.ArgumentParser(); p.add_argument('--topic',default=None); p.add_argument('--difficulty',default='senior'); p.add_argument('--count',type=int,default=5); p.add_argument('--answers',action='store_true'); a=p.parse_args(); qs=select_questions(load_question_bank(ROOT/'config/session25_interview_questions.json'),topic=a.topic,difficulty=a.difficulty,count=a.count); print('SESSION 25 MOCK INTERVIEW');
    for i,q in enumerate(qs,1):
        print(f'\n{i}. [{q.topic}] {q.question}')
        if a.answers: print('Key points:',', '.join(q.key_points)); print('Model answer:',q.model_answer)
if __name__=='__main__':main()
