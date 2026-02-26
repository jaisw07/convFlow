from typing import List, Dict, Optional
from enum import Enum

class CandidateProfile:
    def __init__(self):
        self.name = ""
        self.skills = []
        self.projects = []
        self.experience = []
        self.missing_fields = []

    def update_missing_fields(self):
        missing = []
        if not self.name:
            missing.append("name")
        if not self.skills:
            missing.append("skills")
        if not self.projects:
            missing.append("projects")
        if not self.experience:
            missing.append("experience")

        self.missing_fields = missing

class InterviewPhase(Enum):
    INTRO = "introduction"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLOSING = "closing"

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
        self.phase = InterviewPhase.INTRO
        self.profile = CandidateProfile()
        self.summary = ""
        self.anomaly_flags = {}
        self.profile.update_missing_fields()
        self.phase_turn_count = 0

    def update_phase(self):
        if self.phase == InterviewPhase.INTRO:
            if not self.profile.missing_fields:
                self.phase = InterviewPhase.TECHNICAL
                self.phase_turn_count = 0

        elif self.phase == InterviewPhase.TECHNICAL:
            if self.phase_turn_count >= 4:
                self.phase = InterviewPhase.BEHAVIORAL
                self.phase_turn_count = 0

        elif self.phase == InterviewPhase.BEHAVIORAL:
            if self.phase_turn_count >= 3:
                self.phase = InterviewPhase.CLOSING
                self.phase_turn_count = 0

    def record_turn(self, question: str, answer: str, evaluation: Dict):
        self.turn_count += 1
        self.phase_turn_count += 1
        self.current_question = question

        self.last_evaluation = evaluation

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