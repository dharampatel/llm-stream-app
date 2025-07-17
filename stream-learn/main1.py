# main.py
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import AsyncGenerator
from dotenv import load_dotenv

from app.config import llm

load_dotenv()

app = FastAPI()

# Pydantic input model
class QuestionRequest(BaseModel):
    question: str

prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])
chain = prompt | llm

# Stream response as generator
async def stream_chain(question: str) -> AsyncGenerator[str, None]:
    async for chunk in chain.astream({"question": question}):
        if chunk.content:
            yield chunk.content

@app.post("/stream")
async def stream_answer(payload: QuestionRequest, request: Request):
    return StreamingResponse(stream_chain(payload.question), media_type="text/plain")

@app.post("/chat")
def chat(payload: QuestionRequest):
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | llm

    # Generator to yield chunks
    def generate():
        for chunk in chain.stream({"question": payload.question}):
            yield chunk.content  # only the text content

    return StreamingResponse(generate(), media_type="text/plain")


