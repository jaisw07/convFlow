def create_initial_state():

    return {
        "phase": "intro",
        "question_index": 0,
        "followup_count": 0,

        "last_question": "",
        "last_answer": "",

        "rolling_summary": "",
        "asked_questions_phase": [],

        "candidate_profile": {
            "experience_level": "fresher",
            "technical_score": 0,
            "communication_score": 0,
            "reasoning_score": 0,
            "detected_skills": [],
            "project_topics": []
        }
    }