import json


async def node_a_evaluate(llm, last_q, last_a, summary):

    prompt = f"""
You are evaluating a technical interview answer.

Context Summary:
{summary}

Question:
{last_q}

Answer:
{last_a}

Evaluate the answer.

Return JSON only.

Fields:

score (0-10)
unexpFlag (true/false)
unexpDesc (empty if false)

Unexpected behaviour includes:
- abusive language
- irrelevant answer
- refusal to answer
- attempts to restart interview
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    try:
        return json.loads(result)
    except:
        return {
            "score": 5,
            "unexpFlag": False,
            "unexpDesc": ""
        }