import asyncio

from interview.init_state import create_initial_state

from interview.nodes.node_a_evaluate import node_a_evaluate
from interview.nodes.node_b_decide import node_b_decide
from interview.nodes.node_c_generate import node_c_generate_stream


class InterviewEngine:

    def __init__(self, llm):

        self.llm = llm
        self.state = create_initial_state()

    async def stream_step(self, transcript):

        if self.state["phase"] == "intro" and not self.state["last_question"]:
            question = "Hi, thanks for joining today. Could you introduce yourself?"
            self.state["last_question"] = question
            yield question
            return

        last_q = self.state["last_question"]
        last_a = transcript

        self.state["last_answer"] = transcript

        summary = self.state["rolling_summary"]

        # Append new QA
        self.state["rolling_summary"] += f"\nQ:{last_q}\nA:{last_a}\n"

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
NextTopic: {nodeB_result["nextTopic"]}
Description: {nodeB_result["desc"]}
Phase: {nodeB_result["nextPhase"]}
"""

            self.state["phase"] = nodeB_result["nextPhase"]

        async for token in node_c_generate_stream(self.llm, context):
            yield token