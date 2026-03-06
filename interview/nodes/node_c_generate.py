async def node_c_generate_stream(llm, context):

    prompt = f"""
You are a professional interviewer speaking to a candidate.

Context:
{context}

Generate the next spoken response naturally.
Ask only one question.
"""

    async for token in llm.stream_response(prompt):
        yield token