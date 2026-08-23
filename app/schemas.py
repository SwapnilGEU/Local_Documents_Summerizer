from pydantic import BaseModel, Field, field_validator


class RAGAnswer(BaseModel):
    """Validated shape of the final answer returned by the RAG pipeline.

    Kept intentionally strict: this is what generate_validated_answer()
    checks the raw LLM string against on every attempt.
    """

    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def not_blank_or_placeholder(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("answer is empty after stripping whitespace")

        # Guards against degenerate LLM output (e.g. just punctuation,
        # or the model echoing the prompt tags back).
        if len(cleaned) < 3:
            raise ValueError("answer is too short to be a real response")

        if cleaned.lower() in {"<|assistant|>", "<|system|>", "<|user|>"}:
            raise ValueError("answer is just an echoed prompt tag")

        return cleaned
