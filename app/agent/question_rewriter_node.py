from app.agent.state import AgentState
from app.config import llm


from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def question_rewriter(state: AgentState):
    print(f"Enter question_rewriter")
    # Reset relevant state keys
    state['on_topic'] = ""
    state['documents'] = []
    state['rephrased_question'] = ""
    state['proceed_to_generate'] = False
    state['rephrase_count'] = 0

    # Ensure messages list exists
    if "messages" not in state or state["messages"] is None:
        state['messages'] = []

    # Append current question if not already present
    if not any(msg.content == state['question'].content for msg in state['messages']):
        state['messages'].append(state['question'])

    if len(state['messages']) > 1:
        question = state['question'].content
        raw_conversation = state['messages'][:-1]  # Exclude current question

        # Convert to (role, content) tuples
        conversation = []
        for msg in raw_conversation:
            if isinstance(msg, HumanMessage):
                conversation.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                conversation.append(("assistant", msg.content))
            elif isinstance(msg, SystemMessage):
                conversation.append(("system", msg.content))

        # Limit to the last 10 turns (can be customized)
        limited_conversation = conversation[-10:]

        # Create prompt
        message_template = [
            ("system", "You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval.")
        ] + limited_conversation + [
            ("user", "{question}")
        ]

        chat_prompt = ChatPromptTemplate.from_messages(message_template)
        chain = chat_prompt | llm
        response = chain.invoke({"question": question})
        print(f"Enter question_rewriter: {response.content}")
        state['rephrased_question'] = response.content.strip()
    else:
        print(f"Enter question_rewriter: {state['question'].content}")
        state['rephrased_question'] = state["question"].content

    return state







