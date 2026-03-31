from typing import TypedDict, Dict, Any, List


class InterviewState(TypedDict):
    # Dialogue progress
    phase: str
    question_index: int
    followup_count: int

    # Current turn
    last_question: str
    last_answer: str
    last_assistant_message: str

    # Context
    rolling_summary: str
    asked_questions_phase: List[str]

    # User info
    user_profile: Dict[str, Any]