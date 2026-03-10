import json


async def node_b_decide(
    llm,
    last_q,
    last_a,
    summary,
    asked_questions,
    phase,
    q_index,
    followups
):

    prompt = f"""
You are controlling the flow of a technical interview conversation. Your job is to decide what the interviewer should do next.

Interview Phase:
{phase}

Conversation Summary:
{summary}

Last Question:
{last_q}

Candidate Response:
{last_a}

Questions already asked this phase:
{asked_questions}

Decide interviewer intent.

Possible intents:

acknowledge
followup
answer_candidate_question
clarify
next_topic
repeat_question

Rules:

If candidate asked a question → answer_candidate_question

If candidate asks to repeat the question → repeat_question

If answer incomplete → followup

If answer complete → next_topic

Answer is unclear if:
- nan input
- sentence fragments
- nonsensical phrases
- unrelated words
- very short responses like "uh", "what", "sorry"

If transcript appears corrupted or meaningless → clarify

Return JSON:

intent
reason
topic
nextPhase
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    try:
        return json.loads(result)
    except:
        return {
            "intent": "followup",
            "reason": "model output parsing failed",
            "topic": "clarify previous answer",
            "nextPhase": phase
        }