from typing import AsyncGenerator

from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from starlette.responses import JSONResponse, StreamingResponse

from app.schema import QuestionPayload

router = APIRouter()


@router.post("/ask/")
async def ask_question(payload: QuestionPayload, request: Request):
    config = {"configurable": {"thread_id": payload.thread_id}}
    state = {
        "question": HumanMessage(content=payload.question),
        "messages": []
    }
    result = await request.app.state.graph.ainvoke(state, config)

    # Extract content safely
    content = result["messages"].content if isinstance(result["messages"], BaseMessage) else result["messages"]
    return {"answer": content}


@router.post("/ask-stream/")
async def ask_question_stream(payload: QuestionPayload, request: Request):
    config = {"configurable": {"thread_id": payload.thread_id}}

    async def stream_response() -> AsyncGenerator[str, None]:
        state = {"question": HumanMessage(content=payload.question), "messages": []}

        async for event in request.app.state.graph.astream_events(state, config):

            if event.get("event") == "on_chain_end":
                output = event.get("data", {}).get("output")

                # If output is a Pydantic model like QuestionGrader
                if hasattr(output, "messages"):
                    messages = output.messages
                elif isinstance(output, dict):
                    messages = output.get("messages", [])
                else:
                    messages = []

                if messages and isinstance(messages[-1], AIMessage):
                    print("content1")
                    yield messages[-1].content

            elif event.get("event") == "on_chain_stream":
                chunk = event.get("data", {}).get("chunk", {})
                generate_answer = chunk.get("generate_answer", {})
                messages = generate_answer.get("messages", [])
                if messages and isinstance(messages[-1], AIMessage):
                    print("content2")
                    yield messages[-1].content

    return StreamingResponse(stream_response(), media_type="text/plain")
