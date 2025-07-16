"""
Image Transforms Module

This module defines various transformations for image data. These transformations include
converting grayscale tensors to RGB, applying general image transformations, CIFAR-10-specific
image preprocessing, and resizing images while maintaining their aspect ratio.

Classes:
    - TensorGrayToRgb: Converts grayscale tensors to RGB format.
    - ImageTransform: Applies a series of transformations to an `Image` object.
    - Cifar10ImageTransform: Preprocesses CIFAR-10 images with normalization, padding, and cropping.
    - FixedAspectRatioResize: Resizes images while maintaining their aspect ratio.

Dependencies:
    - typing: For type annotations.
    - numpy: For numerical operations.
    - torch: For tensor operations.
    - PIL.Image: For handling image files.
    - torchvision.transforms: For image transformation utilities.
    - atria_core.logger: For logging utilities.
    - atria_registry: For registering transformations.
    - atria_core.types: For the `Image` class.
    - atria_transforms.core.transforms.base: For the `DataTransform` base class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable

from atria_core.logger import get_logger
from atria_core.transforms import Compose, DataTransform
from atria_core.types import Image

from atria_transforms.registry import DATA_TRANSFORM

logger = get_logger(__name__)


@DATA_TRANSFORM.register("tf_composer")
class TransformsComposer(DataTransform):
    def __init__(
        self,
        transforms: list[Callable] | list[dict] | Callable,
        apply_path: str | None = None,
    ):
        super().__init__(apply_path=apply_path)
        self.transforms = transforms
        self._validate_transforms()

    def _validate_transforms(self):
        if isinstance(self.transforms, list) and all(
            callable(x) for x in self.transforms
        ):
            self.transforms = Compose(self.transforms)
        elif isinstance(self.transforms, list) and all(
            isinstance(x, dict) and "name" in x and "kwargs" in x
            for x in self.transforms
        ):
            self.transforms = Compose(
                [
                    DATA_TRANSFORM.load_from_registry(x["name"], **x.get("kwargs", {}))
                    for x in self.transforms
                ]
            )
        elif callable(self.transforms):
            self.transforms = Compose([self.transforms])
        else:
            raise ValueError(
                "Transforms must be a list of callables or a single callable. Got: "
                f"{self.transforms}"
            )

    def _apply_transforms(self, input: "Image") -> "Image":
        """
        Applies the transformation pipeline to the `Image` object.

        Args:
            input (Image): The input `Image` object.

        Returns:
            Image: The transformed `Image` object.
        """
        input.content = self.transforms(input.content)
        return input


@DATA_TRANSFORM.register("image")
class ImageTransformsComposer(TransformsComposer):
    _REGISTRY_CONFIGS = {
        "default": {
            "transforms": [
                {"name": "image/resize", "kwargs": {"apply_path": "content"}},
                {"name": "image/to_tensor", "kwargs": {"apply_path": "content"}},
                {
                    "name": "image/tensor_gray_to_rgb",
                    "kwargs": {"apply_path": "content"},
                },
                {"name": "image/normalize", "kwargs": {"apply_path": "content"}},
            ]
        }
    }

    def __init__(
        self,
        transforms: list[Callable] | list[dict] | Callable,
        apply_path: str | None = "image",
    ):
        super().__init__(transforms=transforms, apply_path=apply_path)

    def _apply_transforms(self, input: "Image") -> "Image":
        """
        Applies the transformation pipeline to the `Image` object.

        Args:
            input (Image): The input `Image` object.

        Returns:
            Image: The transformed `Image` object.
        """
        return self.transforms(input)
