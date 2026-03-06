from typing import TypedDict, Dict, Any, List


class InterviewState(TypedDict):
    # Interview progress
    phase: str
    question_index: int
    followup_count: int

    # Current turn
    last_question: str
    last_answer: str

    # Context
    rolling_summary: str
    asked_questions_phase: List[str]

    # Candidate info
    candidate_profile: Dict[str, Any]