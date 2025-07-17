import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from app.config import llm

st.title("LangChain Stream Demo")

# Define a simple chain
prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])
chain = prompt | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # Stream the response from the LangChain chain
        response_generator = chain.stream({"question": prompt_input})
        full_response = st.write_stream(response_generator)
    st.session_state.messages.append({"role": "assistant", "content": full_response})