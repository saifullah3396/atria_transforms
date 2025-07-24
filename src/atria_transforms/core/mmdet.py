from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger
from atria_core.transforms import DataTransform
from atria_core.utilities.repr import RepresentationMixin
from atria_registry import RegistryConfig
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from atria_core.types import DocumentInstance


from atria_transforms.registry import DATA_TRANSFORM

logger = get_logger(__name__)


class MMDetInput(BaseModel, RepresentationMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    inputs: list[Any] | Any
    data_samples: DetDataSample


@TRANSFORMS.register_module()
class RandomChoiceResize(BaseTransform):
    def __init__(self, scales: Sequence[int | tuple], **resize_kwargs) -> None:
        super().__init__()

        import mmengine
        from mmdet.datasets.transforms import Resize

        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert mmengine.is_seq_of(self.scales, (tuple, int))
        self.resize = Resize(scale=0, **resize_kwargs)

    @cache_randomness
    def _random_select(self) -> tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        import numpy as np

        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()
        self.resize.scale = target_scale
        results = self.resize(results)
        results["scale_idx"] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scales={self.scales}"
        repr_str += f", resize={self.resize})"
        return repr_str


@DATA_TRANSFORM.register(
    "document_instance_mmdet_transform",
    configs=[
        RegistryConfig(name="train", is_training=True),
        RegistryConfig(name="evaluation", is_training=False),
    ],
)
class DocumentInstanceMMDetTransform(DataTransform):
    train_scale: list[tuple[int, int]] | tuple[int, int] = Field(
        default=(512, 400), description="Scale for training images."
    )
    test_scale: list[tuple[int, int]] | tuple[int, int] = Field(
        default=(512, 400), description="Scale for testing images."
    )
    is_training: bool = Field(
        default=False, description="Whether the transform is used for training."
    )
    use_test_time_augmentation: bool = Field(
        default=False, description="Whether to use test time augmentation."
    )

    def model_post_init(self, context: Any) -> None:
        import torchvision.transforms as T
        from mmdet.datasets.transforms import (
            LoadAnnotations,
            PackDetInputs,
            RandomFlip,
            Resize,
        )

        if self.is_training:
            from mmcv.transforms import RandomChoiceResize, TestTimeAug

            if isinstance(self.train_scale, tuple):
                train_scale = [self.train_scale]
            self._transform = T.Compose(
                [
                    LoadAnnotations(with_bbox=True, with_mask=True),
                    RandomChoiceResize(scales=train_scale, keep_ratio=True),
                    RandomFlip(prob=0.5),
                    PackDetInputs(
                        meta_keys=(
                            "id",
                            "img_id",
                            "img_id",
                            "img_path",
                            "ori_shape",
                            "img_shape",
                            "scale_factor",
                            "flip",
                            "flip_direction",
                        )
                    ),
                ]
            )
        else:
            import torchvision.transforms as T
            from mmcv.transforms import RandomChoiceResize, TestTimeAug

            if self.use_test_time_augmentation:
                if isinstance(self.test_scale, tuple):
                    test_scale = [self.test_scale]
                self._transform = T.Compose(
                    [
                        LoadAnnotations(with_bbox=True, with_mask=True, box_type=None),
                        TestTimeAug(
                            transforms=[
                                [
                                    RandomChoiceResize(
                                        scales=test_scale, keep_ratio=True
                                    )
                                ],
                                [RandomFlip(prob=0.0), RandomFlip(prob=1.0)],
                                [
                                    PackDetInputs(
                                        meta_keys=(
                                            "__key__",
                                            "__index__",
                                            "img_id",
                                            "img_path",
                                            "ori_shape",
                                            "img_shape",
                                            "scale_factor",
                                            "flip",
                                            "flip_direction",
                                        )
                                    )
                                ],
                            ]
                        ),
                    ]
                )
            else:
                import torchvision.transforms as T
                from mmcv.transforms import RandomChoiceResize, TestTimeAug

                if isinstance(test_scale, list):
                    test_scale = test_scale[0]
                self._transform = T.Compose(
                    [
                        LoadAnnotations(with_bbox=True, with_mask=True),
                        Resize(scale=test_scale, keep_ratio=True),
                        PackDetInputs(
                            meta_keys=(
                                "__key__",
                                "__index__",
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            )
                        ),
                    ]
                )

    def _prepare_instances(self, document_instance: DocumentInstance):
        from atria_core.types import BoundingBoxMode

        assert document_instance.gt.layout.annotated_objects is not None, (
            f"Document instance must have annotated objects in the ground truth for {self.__class__} ."
        )

        # generate instances in mmdet format
        instances = []
        for ann in document_instance.gt.layout.annotated_objects:
            if not ann.bbox.mode == BoundingBoxMode.XYXY:
                ann.bbox.switch_mode()
            instance = {
                "bbox": ann.bbox.value,
                "bbox_label": ann.label.value,
                "ignore_flag": 1 if ann.iscrowd else 0,
            }
            if ann.segmentation is not None:
                instance["mask"] = ann.segmentation
            instances.append(instance)

        return instances

    def _apply_transforms(self, document_instance: DocumentInstance) -> MMDetInput:
        import numpy as np

        instances = self._prepare_instances(document_instance)
        return MMDetInput(
            **self._transform(
                {
                    "id": document_instance.id,
                    "img_id": document_instance.index,
                    "instances": instances,
                    "img": np.array(document_instance.image.content),
                    "img_shape": (
                        document_instance.image.height,
                        document_instance.image.width,
                    ),
                    "ori_shape": (
                        document_instance.image.source_size[1],
                        document_instance.image.source_size[0],
                    ),
                }
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  is_training={self.is_training},\n"
            f"  use_test_time_augmentation={self.use_test_time_augmentation},\n"
            f"  target_key={self.key},\n"
            f"  transform={self._transform},\n"
            f")"
        )
