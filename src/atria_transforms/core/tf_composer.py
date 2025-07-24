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

from typing import Any

from atria_core.logger import get_logger
from atria_core.transforms import Compose, DataTransform
from atria_core.types import Image
from atria_registry.registry_config import RegistryConfig

from atria_transforms.registry import DATA_TRANSFORM

logger = get_logger(__name__)


@DATA_TRANSFORM.register("tf_composer")
class TransformsComposer(DataTransform):
    configs: list[RegistryConfig]
    _transforms: Compose | None = None

    def model_post_init(self, context: Any) -> None:
        self._initialize_transforms()

    def _initialize_transforms(self):
        self._transforms = Compose(
            [
                DATA_TRANSFORM.load_from_registry(config.name, **config.model_extra)
                for config in self.configs
            ]
        )

    def _apply_transforms(self, input: "Image") -> "Image":
        """
        Applies the transformation pipeline to the `Image` object.

        Args:
            input (Image): The input `Image` object.

        Returns:
            Image: The transformed `Image` object.
        """
        input.content = self._transforms(input.content)
        return input


@DATA_TRANSFORM.register(
    "image",
    configs=[
        RegistryConfig(
            name="default",
            configs=[
                RegistryConfig(name="image/resize", apply_path="content"),
                RegistryConfig(name="image/to_tensor", apply_path="content"),
                RegistryConfig(name="image/tensor_gray_to_rgb", apply_path="content"),
                RegistryConfig(name="image/normalize", apply_path="content"),
            ],
        )
    ],
)
class ImageTransformsComposer(TransformsComposer):
    apply_path: str | None = "image"

    def _apply_transforms(self, input: "Image") -> "Image":
        """
        Applies the transformation pipeline to the `Image` object.

        Args:
            input (Image): The input `Image` object.

        Returns:
            Image: The transformed `Image` object.
        """
        return self.config(input)
