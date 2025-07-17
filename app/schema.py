from pydantic import BaseModel, Field


class QuestionPayload(BaseModel):
    question: str
    thread_id: str = Field(default="001")
