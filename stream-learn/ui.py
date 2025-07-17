# ui.py (Streamlit)
import time

import streamlit as st
import httpx

API_URL = "http://localhost:8000/chat"

st.title("LLM Chat with Streaming via API")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt_input := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # Stream from API using httpx
        def stream_from_api():
            with httpx.stream("POST", API_URL, json={"question": prompt_input}, timeout=60.0) as response:
                for chunk in response.iter_text():
                    time.sleep(0.50)
                    yield chunk

        full_response = st.write_stream(stream_from_api())
    st.session_state.messages.append({"role": "assistant", "content": full_response})
