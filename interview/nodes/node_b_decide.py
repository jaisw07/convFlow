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
You are planning the next question in a technical interview.

Phase: {phase}

Rolling Summary:
{summary}

Last Question:
{last_q}

Last Answer:
{last_a}

Questions already asked this phase:
{asked_questions}

Return JSON:

nextTopic
desc
nextPhase

Rules:
- If answer incomplete -> followup
- Avoid repeating topics
- Suitable for fresher
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    try:
        return json.loads(result)
    except:
        return {
            "nextTopic": "followup",
            "desc": "ask candidate to elaborate",
            "nextPhase": phase
        }