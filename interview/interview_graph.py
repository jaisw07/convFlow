from langgraph.graph import StateGraph, END

from interview.nodes.evaluate_answer import evaluate_answer
from interview.nodes.update_profile import update_profile
from interview.nodes.decision_logic import decision_logic
from interview.nodes.generate_question import generate_question

from interview.state import InterviewState


def build_graph(llm):

    graph = StateGraph(InterviewState)

    async def evaluate_node(state):
        return await evaluate_answer(state, llm)

    async def generate_node(state):
        return await generate_question(state, llm)

    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("profile", update_profile)
    graph.add_node("decide", decision_logic)
    
    graph.set_entry_point("evaluate")

    graph.add_edge("evaluate", "profile")
    graph.add_edge("profile", "decide")
    graph.add_edge("decide", "generate")
    graph.add_edge("generate", END)

    return graph.compile()