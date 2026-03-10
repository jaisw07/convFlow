async def node_c_generate_stream(llm, context):

    prompt = f"""
    You are a professional technical interviewer.

Follow the conversation intent exactly.

Intent meanings:

acknowledge
→ briefly acknowledge the candidate response.

followup
→ ask a deeper question about the same topic.

clarify
→ ask candidate to clarify their response.

answer_candidate_question
→ answer the candidate's question briefly then continue interview.

next_topic
→ move interview forward naturally.

repeat_question
→ repeat the previous question clearly

Conversation context:
{context}

Generate the next spoken interviewer response.

Speak naturally.
Ask only one question.
"""

    async for token in llm.stream_response(prompt):
        yield token