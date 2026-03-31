async def rolling_summarize(llm, transcript):

    prompt = f"""
You are compressing a Type 1 Diabetes support conversation summary.

Summarize the conversation while preserving:

- inferred user type (parent or child_patient) and confidence clues
- user concerns and open questions
- emotional tone and support needs
- safety or escalation cues discussed
- key profile facts (name, child age, diagnosis context, care challenges)
- myths already corrected and guidance already given

Transcript:

{transcript}

Return a concise plain text summary.
"""

    result = ""

    async for token in llm.stream_response(prompt):
        result += token

    return result.strip()