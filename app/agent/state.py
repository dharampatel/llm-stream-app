from typing import TypedDict, Annotated, List

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    on_topic:str
    rephrased_question: str
    documents: List[Document]
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage
