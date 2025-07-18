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

#async stream implementation
@app.post("/chat")
async def chat(payload: QuestionRequest):
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | llm

    # Generator to yield chunks
    async def generate():
        async for chunk in chain.astream({"question": payload.question}):
            yield chunk.content  # only the text content
    # we can use either 'text/plain' or 'text/event-stream', but for streaming 'text/event-stream' recommended
    return StreamingResponse(generate(), media_type="text/event-stream")


#sync stream implementation
@app.post("/chat1")
def chat(payload: QuestionRequest):
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | llm

    # Generator to yield chunks
    def generate():
        for chunk in chain.stream({"question": payload.question}):
            yield chunk.content  # only the text content

    return StreamingResponse(generate(), media_type="text/plain")


