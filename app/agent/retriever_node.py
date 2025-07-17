from pydantic import BaseModel, Field

from app.config import get_retriever
from app.agent.state import AgentState
from app.config import llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage


template = """Answer the question based on the following context and the Chat history. Especially take the latest question into consideration:

    Chat history: {history}
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm


async def retrieve(state: AgentState):
    print(f"Enter retrieve")
    documents = await get_retriever().ainvoke(state["rephrased_question"])
    state["documents"] = documents
    return state


class DocumentGrader(BaseModel):
    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
    )


async def retrieval_classifier(state: AgentState):
    print("Entering retrieval_grader")
    system_message = SystemMessage(
        content="""You are a grader assessing the relevance of a retrieved document to a user question.
    Only answer with 'Yes' or 'No'.
    
    If the document contains information relevant to the user's question, respond with 'Yes'.
    Otherwise, respond with 'No'."""
    )

    structured_llm = llm.with_structured_output(DocumentGrader)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = await grader_llm.ainvoke({})
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)

    print(f"Enter retrieve: {relevant_docs}")
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    return state


def proceed_router(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    print(f"Enter proceed_router rephrase_count: {rephrase_count}, proceed_to_generate: {state.get('proceed_to_generate', False)}")
    if state.get("proceed_to_generate", False):
        return "generate_answer"
    elif rephrase_count >= 2:
        return "cannot_answer"
    else:
        return "refine_question"


async def refine_question(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    print(f"Enter refine_question rephrase_count: {rephrase_count}")
    if rephrase_count >= 2:
        return state

    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
    Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    prompt = refine_prompt.format()
    response = await llm.ainvoke(prompt)
    refined_question = response.content.strip()
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    print(f"Enter refine_question refined_question: {refined_question}")
    return state


async def generate_answer(state: AgentState):
    print(f"Enter generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = await rag_chain.ainvoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()
    print(f"Enter generate_answer generation: {generation}")
    state["messages"].append(AIMessage(content=generation))
    return state


def cannot_answer(state: AgentState):
    print(f"Enter cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state