from app.agent.state import AgentState
from app.config import llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

class QuestionGrader(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

def question_classifier(state: AgentState):
    print("Enter question_classifier")
    system_message = SystemMessage(
        content="""
    You are a smart classifier that determines whether a user's question is related to **financial documents and reports** for the fiscal years **2024 or 2025**. The topics may include but are not limited to:
    
    1. Financial statements or summaries (e.g. income statement, balance sheet, cash flow)
    2. Tax reports or filings
    3. Company annual reports for 2024 or 2025
    4. Revenue, expense, or profit analysis
    5. Investment or budget planning documents
    6. Audit findings or financial compliance
    7. Regulatory or legal financial disclosures for 2024–2025
    
    If the question is about **any of these** or **closely related financial topics from those years**, respond with **'Yes'**.
    
    If the question is **not related** to those topics (e.g. personal finance, unrelated business inquiries, general non-financial queries), respond with **'No'**.
    """
    )

    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )

    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    structured_llm = llm.with_structured_output(QuestionGrader)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    print(f"Enter question_classifier: {result.score.strip()}")
    state["on_topic"] = result.score.strip()
    return state


def on_topic_router(state: AgentState):
    on_topic = state.get("on_topic", "").strip().lower()
    print(f"Enter on_topic_router: {on_topic}")
    if on_topic == "yes":
        return "retrieve"
    else:
        return "off_topic_response"


def off_topic_response(state: AgentState):
    print(f"Enter off_topic_response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state

