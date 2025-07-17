import streamlit as st
import asyncio
from langchain_core.prompts import ChatPromptTemplate

import os
from  dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

st.title("LangChain Async Demo")

prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])
chain = prompt | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

async def main():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                response = await chain.ainvoke({"question": prompt_input})
                st.write(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})

# 🚀 Run the async main function
asyncio.run(main())
