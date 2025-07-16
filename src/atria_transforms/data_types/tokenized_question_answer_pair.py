from typing import Self

import torch
from atria_core.types import QuestionAnswerPair
from pydantic import model_validator


class TokenizedQuestionAnswerPair(QuestionAnswerPair):
    answer_starts: torch.Tensor
    answer_ends: torch.Tensor

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        # this type is by default initialized as a tensor and has no raw representation
        self._is_tensor = True

        assert len(self.answer_starts) == len(self.answer_ends), (
            "tokenized_answer_starts and tokenized_answer_ends must have the same length"
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
