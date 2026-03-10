import asyncio

from interview.init_state import create_initial_state

from interview.nodes.node_a_evaluate import node_a_evaluate
from interview.nodes.node_b_decide import node_b_decide
from interview.nodes.node_c_generate import node_c_generate_stream
from interview.nodes.rolling_summarizer import rolling_summarize
from interview.utils.summary_utils import summary_trigger


class InterviewEngine:

    def __init__(self, llm):

        self.llm = llm
        self.state = create_initial_state()
    
    async def background_summarize(self):
        summary = self.state["rolling_summary"]
        new_summary = await rolling_summarize(self.llm, summary)
        self.state["rolling_summary"] = new_summary
        print("🧠 Rolling summary compressed.")

    async def stream_step(self, transcript):

        if self.state["phase"] == "intro" and not self.state["last_question"]:
            question = "Hi, thanks for joining today. Could you introduce yourself?"
            self.state["last_question"] = question
            yield question
            return

        last_q = self.state["last_question"]
        last_a = transcript

        self.state["last_answer"] = transcript

        # Append new QA
        self.state["rolling_summary"] += f"\nQ:{last_q}\nA:{last_a}\n"

        summary = self.state["rolling_summary"]

        # Trigger background summarization
        if summary_trigger(self.state["rolling_summary"]):
            asyncio.create_task(self.background_summarize())

        # Run nodes in parallel
        nodeA = asyncio.create_task(
            node_a_evaluate(self.llm, last_q, last_a, summary)
        )

        nodeB = asyncio.create_task(
            node_b_decide(
                self.llm,
                last_q,
                last_a,
                summary,
                self.state["asked_questions_phase"],
                self.state["phase"],
                self.state["question_index"],
                self.state["followup_count"],
            )
        )

        nodeA_result, nodeB_result = await asyncio.gather(nodeA, nodeB)

        if nodeA_result["unexpFlag"]:

            context = f"""
Unexpected behaviour detected.

Description:
{nodeA_result["unexpDesc"]}

Respond professionally and redirect interview.
"""

        else:

            context = f"""
Interview Phase: {nodeB_result["nextPhase"]}

Intent: {nodeB_result["intent"]}

Reason: {nodeB_result["reason"]}

Topic: {nodeB_result["topic"]}

Last question: {last_q}

Candidate said: {last_a}
"""

            self.state["phase"] = nodeB_result["nextPhase"]

        question_buffer = ""

        async for token in node_c_generate_stream(self.llm, context):
            question_buffer += token
            yield token

        self.state["last_question"] = question_buffer.strip()
        self.state["asked_questions_phase"].append(self.state["last_question"])
