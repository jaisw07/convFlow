def create_initial_state():

    return {
        "phase": "onboarding",
        "question_index": 0,
        "followup_count": 0,

        "last_question": "",
        "last_answer": "",
        "last_assistant_message": "",

        "rolling_summary": "",
        "asked_questions_phase": [],

        "user_profile": {
            "user_type": "unknown",  # parent | child_patient | unknown
            "profile_confidence": 0.0,
            "input_language": "unknown",  # english | hindi | hinglish | unknown
            "name": "",
            "child_name": "",
            "child_age": "",
            "diagnosis_context": "",
            "primary_concerns": [],
            "emotional_state": "unknown",
            "medical_team_connected": "unknown",
        }
    }