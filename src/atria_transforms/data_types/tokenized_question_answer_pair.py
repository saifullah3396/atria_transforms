from typing import Self

import torch
from atria_core.types import QuestionAnswerPair
from pydantic import model_validator


class TokenizedQuestionAnswerPair(QuestionAnswerPair):
    _batch_tensor_stack_skip_fields = ["tokenized_answer_start", "tokenized_answer_end"]
    tokenized_answer_start: torch.Tensor | None = None
    tokenized_answer_end: torch.Tensor | None = None

    @classmethod
    def from_qa_pair(cls, qa_pair: QuestionAnswerPair) -> Self:
        """Convert a QuestionAnswerPair to a TokenizedQuestionAnswerPair."""
        return cls(
            id=qa_pair.id,
            question_text=qa_pair.question_text,
            answer_start=qa_pair.answer_start,
            answer_end=qa_pair.answer_end,
            answer_text=qa_pair.answer_text,
        )

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        if self.answer_start is not None and self.answer_end is not None:
            assert len(self.answer_start) == len(self.answer_end), (
                "answer_start and answer_end must have the same length"
            )
        return self

    def load(self) -> Self:
        raise NotImplementedError(
            "TokenizedQuestionAnswerPair does not support loading."
        )

    def unload(self) -> Self:
        raise NotImplementedError(
            "TokenizedQuestionAnswerPair does not support unloading."
        )

    def to_raw(self) -> Self:
        raise NotImplementedError(
            "TokenizedQuestionAnswerPair does not support converting to raw format."
        )
