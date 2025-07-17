import gradio as gr
import httpx
import asyncio

API_URL = "http://localhost:8000/ask-stream/"  # Your FastAPI stream endpoint

# Async stream fetcher
async def fetch_stream(question: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", API_URL, json={"question": question, "thread_id": "test_thread"}) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    yield line

# Streaming chat function
async def chat_function(message, _):
    partial = ""
    async for chunk in fetch_stream(message):
        partial += chunk
        yield [{"role": "assistant", "content": partial}]

# Launch Chat UI
gr.ChatInterface(fn=chat_function, type="messages", title="📄 RAG Assistant").launch()
