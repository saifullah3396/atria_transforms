"""
Tokenized Object Data Instance Module

This module defines the `TokenizedObjectInstance` and `BatchedTokenizedObjectInstance` classes, which represent data instances
containing tokenized information. These classes are designed to handle tokenized data such as token IDs, word IDs, token labels,
and other related attributes. They are useful for tasks like token classification, sequence labeling, and object detection in
tokenized formats.

Classes:
    - TokenizedObjectInstance: Represents a single tokenized data instance.
    - BatchedTokenizedObjectInstance: Represents a batch of tokenized data instances.

Dependencies:
    - pydantic: For data validation and serialization.
    - atria_core.data_types.data_instance.base: For the base data instance class.
    - atria_core.data_types.generic.image: For handling image data.
    - atria_core.data_types.generic.label: For handling label data.
    - atria_core.data_types.typing.tensor: For defining tensor types.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from types import NoneType
from typing import ClassVar, Self

import torch
from atria_core.types import BaseDataInstance, Image, Label
from atria_core.utilities.common import _rgetattr
from pydantic import ConfigDict, PrivateAttr, model_validator

from atria_transforms.data_types.tokenized_question_answer_pair import (
    TokenizedQuestionAnswerPair,
)


class TokenizedDocumentInstance(BaseDataInstance):
    """
    Represents a single tokenized data instance.

    This class is designed to handle tokenized data such as token IDs, word IDs,
    token labels, and other related attributes. It is useful for tasks like token
    classification, sequence labeling, and object detection in tokenized formats.

    Attributes:
        token_ids (PydanticTensor): The token IDs for the instance.
        token_bboxes (PydanticTensor, optional): The bounding boxes for the tokens.
        token_type_ids (PydanticTensor, optional): The type IDs for the tokens.
        token_labels (PydanticTensor, optional): The labels for the tokens.
        attention_mask (PydanticTensor, optional): The attention mask for the tokens.
        word_ids (PydanticTensor, optional): The word IDs for the tokens.
        sequence_ids (PydanticTensor, optional): The sequence IDs for the tokens.
        overflow_to_sample_mapping (PydanticTensor, optional): Mapping from overflowed
            tokens to sample indices.
    """

    model_config = ConfigDict(validate_assignment=False)
    _batch_skip_fields: ClassVar[list[str]] = ["ocr", "page_id", "total_num_pages"]
    _batch_tensor_stack_skip_fields: ClassVar[list[str]] = [
        "token_ids",
        "token_bboxes",
        "token_labels",
        "attention_mask",
        "word_ids",
        "sequence_ids",
        "overflow_to_sample_mapping",
    ]
    _tokenizer = PrivateAttr(default=None)

    token_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor
    word_ids: torch.Tensor
    sequence_ids: torch.Tensor
    overflow_to_sample_mapping: torch.Tensor
    image: Image | None = None
    label: Label | None = None
    words: list[str] | None = None
    qa_pair: TokenizedQuestionAnswerPair | None = None

    @property
    def prediction_indices_mask(self) -> torch.Tensor:
        prediction_indices_mask = torch.zeros_like(
            self.token_ids, dtype=torch.bool, device=self.token_ids.device
        )
        for idx, word_ids_per_sample in enumerate(self.word_ids):
            if self.token_labels is not None:
                prediction_indices = [
                    i
                    for i in range(len(word_ids_per_sample))
                    if word_ids_per_sample[i] != word_ids_per_sample[i - 1]
                    and self.token_labels[idx][i] != -100
                ]
            else:
                prediction_indices = [
                    i
                    for i in range(len(word_ids_per_sample))
                    if word_ids_per_sample[i] != word_ids_per_sample[i - 1]
                ]
            prediction_indices_mask[idx][prediction_indices] = True
        return prediction_indices_mask

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

    @property
    def tokenizer(self):
        """
        Returns the tokenizer associated with this instance.
        If the tokenizer is not set, it raises an error.
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set for this instance.")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        """
        Sets the tokenizer for this instance.
        This is useful for decoding token IDs back to text.
        """
        self._tokenizer = tokenizer

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "TokenizedDocumentInstance":
        """
        Validates the shapes of the tensors in the instance.

        This method checks that the shapes of the token IDs, word IDs, and other
        tensors are consistent with each other. It raises a ValueError if any
        inconsistencies are found.

        Returns:
            TokenizedObjectInstance: The validated instance.

        Raises:
            AssertionError: If the tensor shapes are inconsistent.
        """
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key in [
                "token_ids",
                "word_ids",
                "token_labels",
                "token_type_ids",
                "attention_mask",
            ]:
                assert value.ndim == 2, (
                    f"Expected 1D tensor, but got {value.ndim}D tensor for {key}."
                )
                assert value.shape == self.token_ids.shape, (
                    f"{key} must have the same shape as token_ids {self.token_ids.shape}."
                )
            if key == "token_bboxes":
                assert value.ndim == 3, (
                    f"Expected 2D tensor, but got {value.ndim}D tensor for {key}."
                )
                assert (
                    value.shape[1] == self.token_ids.shape[1] and value.shape[2] == 4
                ), (
                    f"{key} must have compatible shape with token_ids {self.token_ids.shape}."
                )

        return self

    def select_all_overflow_samples(self) -> tuple[bool, list[int], list[str]]:
        """
        Concatenates all overflowed samples into a single tensor.
        This method is useful for handling overflowed tokens in tokenized data.
        It ensures that the overflowed tokens are properly mapped to their
        corresponding samples.

        If token_ids has a list of tensors with shapes: [(1, 512), (2, 512)],
        This function will concatenate them into a single tensor with shape (3, 512).

        Other elements such as 'image', 'id', etc will be repeated accordingly.
        If images is a list of tensors with shapes: [(1, 3, 224, 224), (1, 3, 224, 224)],
        This function will concatenate them into a single tensor with shape (3, 3, 224, 224) with second image
        repeated twice. This is done for all elments in data instance that are not in 'non_repeated_keys'.

        Args:
            None
        Returns:
            Tuple[bool, list[int], list[str]]: A tuple containing:
                - A boolean indicating whether the concatenation was successful.
                - A list of indices indicating the number of times each sample was repeated.
                - A list of keys that were not repeated.
        """
        assert self._is_batched, (
            "This function only supports batched document instances. Call batched() first."
        )
        repeat_indices = [sample.shape[0] for sample in self.token_ids]

        # we concatenate all lists of overflowed samples into a single tensor
        def _cat_tensor_fields(
            samples_list: list[torch.Tensor],
        ) -> torch.Tensor | None:
            if samples_list is not None:
                return torch.cat(samples_list, dim=0)
            return None

        self.token_ids = _cat_tensor_fields(self.token_ids)
        self.token_bboxes = _cat_tensor_fields(self.token_bboxes)
        self.token_labels = _cat_tensor_fields(self.token_labels)
        self.attention_mask = _cat_tensor_fields(self.attention_mask)
        self.word_ids = _cat_tensor_fields(self.word_ids)
        self.sequence_ids = _cat_tensor_fields(self.sequence_ids)
        self.overflow_to_sample_mapping = _cat_tensor_fields(
            self.overflow_to_sample_mapping
        )
        if self.qa_pair is not None:
            self.qa_pair.answer_starts = _cat_tensor_fields(self.qa_pair.answer_starts)
            self.qa_pair.answer_ends = _cat_tensor_fields(self.qa_pair.answer_ends)

        # we recursively repeat all the remaining fields that are not in 'non_repeated_keys'
        # we recursively repeat all the batched samples with given indices
        self.repeat(
            repeat_indices,
            [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
                "answer_starts",
                "answer_ends",
            ],
        )
        return (
            True,
            repeat_indices,
            [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
                "answer_starts",
                "answer_ends",
            ],
        )

    def select_random_overflow_samples(self):
        """
        Unlike concat_all_overflow_samples, this function randomly selects one sample
        from each overflowed sample and concatenates them into a single tensor.
        """
        assert self._is_batched, (
            "This function only supports batched document instances. Call batched() first."
        )
        assert isinstance(self.token_ids, list), "token_ids must be a list."
        random_select_ids = [
            torch.randint(0, sample.shape[0], size=(1,)).item()
            for sample in self.token_ids
        ]

        def stack_fn(list_of_samples):
            return torch.stack(
                [
                    sample[idx]
                    for sample, idx in zip(
                        list_of_samples, random_select_ids, strict=True
                    )
                ]
            )

        for key in NON_REPEATED_KEYS:
            assert isinstance(_rgetattr(self, key), list | NoneType), (
                f"{key} must be a list."
            )
            _apply(self, key, stack_fn)

    def select_first_overflow_samples(self):
        """
        Unlike concat_all_overflow_samples, this function selects the first sample
        from each overflowed sample and concatenates them into a single tensor.
        """
        assert self._is_batched, (
            "This function only supports batched document instances. Call batched() first."
        )

        def _gather_first_from_sequence_list(
            samples_list: list[torch.Tensor],
        ) -> torch.Tensor | None:
            if samples_list is not None:
                return torch.stack([sample[0] for sample in samples_list])
            return None

        self.token_ids = _gather_first_from_sequence_list(self.token_ids)
        self.token_bboxes = _gather_first_from_sequence_list(self.token_bboxes)
        self.token_labels = _gather_first_from_sequence_list(self.token_labels)
        self.attention_mask = _gather_first_from_sequence_list(self.attention_mask)
        self.word_ids = _gather_first_from_sequence_list(self.word_ids)
        self.sequence_ids = _gather_first_from_sequence_list(self.sequence_ids)
        self.overflow_to_sample_mapping = _gather_first_from_sequence_list(
            self.overflow_to_sample_mapping
        )
        if self.qa_pair is not None:
            self.qa_pair.answer_starts = _gather_first_from_sequence_list(
                self.qa_pair.answer_starts
            )
            self.qa_pair.answer_ends = _gather_first_from_sequence_list(
                self.qa_pair.answer_ends
            )

        self._assert_all_batch_size_equal()

    def _assert_all_batch_size_equal(self):
        """
        Asserts that all fields in the instance have the same batch size.
        This is important for ensuring consistency in batched operations.
        """
        batch_size = self.token_ids.shape[0]
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == batch_size, (
                    f"Field {key} has batch size {value.shape[0]}, expected {batch_size}."
                )

    def decode_tokens(self, token_ids: torch.Tensor):
        assert self._tokenizer is not None, (
            "Tokenizer is not set. Please set the tokenizer before decoding tokens."
        )
        return self._tokenizer.decode(token_ids)
