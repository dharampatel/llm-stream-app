import streamlit as st
import asyncio
from langchain_core.prompts import ChatPromptTemplate

import os
from  dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

st.title("Streamlit+LLM With async stream")

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
        # Async streaming from chain
        async def stream_and_display():
            full_response = ""
            async for chunk in chain.astream({"question": prompt_input}):
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        # write_stream handles async generator
        st.write_stream(stream_and_display)
