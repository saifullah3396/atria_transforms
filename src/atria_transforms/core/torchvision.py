from typing import TYPE_CHECKING, Any

from atria_core.transforms.base import DataTransform
from atria_core.utilities.imports import _resolve_module_from_path
from pydantic import ConfigDict

from atria_transforms.registry import DATA_TRANSFORM

if TYPE_CHECKING:
    import torch


class TorchvisionTransform(DataTransform):  # or inherit from DataTransform if needed
    """
    A wrapper class for applying a specified transformation to a PyTorch tensor.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="allow"
    )

    transform: str  # Name of the torchvision transform class, like "Resize", "ToTensor"
    kwargs: dict[str, Any] = {}

    def model_post_init(self, context: Any) -> None:
        self._transform = _resolve_module_from_path(
            f"torchvision.transforms.{self.transform}"
        )(**self.kwargs)

    @property
    def name(self) -> str:
        return self.transform

    def _apply_transforms(self, image: "torch.Tensor") -> "torch.Tensor":
        return self._transform(image)


# Registering torchvision transformations into the DATA_TRANSFORM registry
DATA_TRANSFORM.register_torchvision_transform(
    "Resize", interpolation=2, size=(224, 224)
)
DATA_TRANSFORM.register_torchvision_transform("ToTensor")
DATA_TRANSFORM.register_torchvision_transform("CenterCrop")
DATA_TRANSFORM.register_torchvision_transform(
    "Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
DATA_TRANSFORM.register_torchvision_transform("RandomHorizontalFlip")
