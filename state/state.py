from typing import List, Dict, Optional


class InterviewState:
    def __init__(self, topic: str = "General Technical Interview"):
        self.topic = topic
        self.turn_count = 0
        self.difficulty = 1
        self.score_history: List[float] = []
        self.current_question: Optional[str] = None
        self.history: List[Dict] = []
        self.last_evaluation = None
        self.followup_stack: List[str] = []


    def record_turn(self, question: str, answer: str, evaluation: Dict):
        self.turn_count += 1
        self.current_question = question

        self.last_evaluation = evaluation

        score = evaluation.get("score", 0.0)
        self.score_history.append(score)

        # Compute composite score
        correctness = evaluation.get("correctness", 5)
        depth = evaluation.get("technical_depth", 5)
        clarity = evaluation.get("clarity", 5)
        confidence = evaluation.get("confidence", 5)

        composite = (
            0.4 * correctness +
            0.3 * depth +
            0.2 * clarity +
            0.1 * confidence
        ) / 10.0  # normalize to 0-1

        self.score_history.append(composite)
        followups = evaluation.get("followup_opportunities", [])
        if followups:
            self.followup_stack.extend(followups)

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