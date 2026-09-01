from dataclasses import dataclass
@dataclass(frozen=True)
class InterviewQuestion: id:str; topic:str; difficulty:str; question:str; key_points:tuple[str,...]; model_answer:str
@dataclass(frozen=True)
class AnswerScore: question_id:str; matched_points:tuple[str,...]; missing_points:tuple[str,...]; score:float
