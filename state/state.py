from typing import List, Dict, Optional


class InterviewState:
    def __init__(self, topic: str = "General Technical Interview"):
        self.topic = topic
        self.turn_count = 0
        self.difficulty = 1
        self.score_history: List[float] = []
        self.current_question: Optional[str] = None
        self.history: List[Dict] = []

    def record_turn(self, question: str, answer: str, evaluation: Dict):
        self.turn_count += 1
        self.current_question = question

        score = evaluation.get("score", 0.0)
        self.score_history.append(score)

        self.history.append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

    def average_score(self) -> float:
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    def adjust_difficulty(self):
        avg = self.average_score()

        if avg > 0.75:
            self.difficulty += 1
        elif avg < 0.4 and self.difficulty > 1:
            self.difficulty -= 1