from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.agent.question_classifier_node import question_classifier, off_topic_response, on_topic_router
from app.agent.question_rewriter_node import question_rewriter
from app.agent.retriever_node import retrieve, retrieval_classifier, generate_answer, refine_question, cannot_answer, \
    proceed_router
from app.agent.state import AgentState


def get_graph():
    checkpointer = MemorySaver()
    workflow = StateGraph(AgentState)
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_classifier", retrieval_classifier)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("cannot_answer", cannot_answer)

    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges(
        "question_classifier",
        on_topic_router,
        {
            "retrieve": "retrieve",
            "off_topic_response": "off_topic_response",
        },
    )
    workflow.add_edge("retrieve", "retrieval_classifier")
    workflow.add_conditional_edges(
        "retrieval_classifier",
        proceed_router,
        {
            "generate_answer": "generate_answer",
            "refine_question": "refine_question",
            "cannot_answer": "cannot_answer",
        },
    )
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("cannot_answer", END)
    workflow.add_edge("off_topic_response", END)
    workflow.set_entry_point("question_rewriter")
    return workflow.compile(checkpointer=checkpointer)