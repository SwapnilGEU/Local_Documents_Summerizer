from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str = Field(min_length=1)