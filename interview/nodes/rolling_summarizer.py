async def rolling_summarize(llm, transcript):

    prompt = f"""
You are compressing an interview transcript summary.

Summarize the conversation while preserving:

- skills demonstrated
- topics discussed
- candidate strengths
- weaknesses
- important technical details

Transcript:

{transcript}

Return a concise summary.
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    return result.strip()