import json
import re


async def node_b_decide(
    llm,
    last_q,
    last_a,
    summary,
    current_user_type,
    current_phase,
    user_profile,
):

    prompt = f"""
You are Node B in a Type 1 Diabetes support assistant.
You must classify user type, detect input language, pick next intent, and suggest profile updates.
The assistant always outputs plain English text in short spoken-friendly style.

Conversation summary:
{summary}

Current user_type:
{current_user_type}

Current phase:
{current_phase}

Current user profile:
{json.dumps(user_profile, ensure_ascii=True)}

Assistant's last message:
{last_q}

User said:
{last_a}

Return strict JSON only:
user_type: one of [child_patient, parent, unknown]
input_language: one of [english, hindi, hinglish, unknown]
language_style: one of [very_simple, simple, standard_clinical]
confidence: float between 0 and 1
phase: one of [onboarding, education, myth_busting, care_guidance, emotional_support, escalation]
intent: one of [ask_profile, answer_question, myth_bust, care_tip, emotional_support, safety_escalation, ask_to_repeat, repeat_last]
topic: short string
reason: short string
understood: true or false
profile_updates: object with optional keys from [name, child_name, child_age, diagnosis_context, emotional_state, medical_team_connected, primary_concerns]

If user indicates their child has T1D -> parent.
If user indicates they are a child with T1D -> child_patient.
If unclear, keep existing user_type when available.
If text is empty, gibberish, only filler, or seems unrecognized ASR -> understood=false and intent=ask_to_repeat.
If user asks to repeat previous assistant statement -> intent=repeat_last.
If user asks emergency danger signs or severe symptoms -> phase=escalation and intent=safety_escalation.
Use language_style=very_simple for child_patient, simple for parent, standard_clinical only when explicitly requested.
Do not invent medical facts in topic/reason.
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    try:
        clean = re.search(r"\{.*\}", result, re.DOTALL)
        if clean:
            return json.loads(clean.group(0))
    except Exception:
        pass

    return {
        "user_type": "unknown",
        "input_language": "unknown",
        "language_style": "simple",
        "confidence": 0.0,
        "phase": "onboarding",
        "intent": "ask_to_repeat",
        "topic": "clarification",
        "reason": "model output parsing failed",
        "understood": False,
        "profile_updates": {},
    }