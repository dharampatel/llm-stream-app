# Streamlit and llm together
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from  dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

st.title("Streamlit+LLM With sync invoke")

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
        with st.spinner(text="Loading..."):
            response = chain.invoke({"question": prompt_input})
            st.write(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
