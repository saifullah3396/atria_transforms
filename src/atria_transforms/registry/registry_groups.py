from atria_registry import RegistryGroup


class DataTransformRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing data transformations.

    This class provides additional methods for registering and managing data
    transformations within the registry system.
    """

    def register_torchvision_transform(self, transform: str, **kwargs):
        """
        Register a torchvision transform.

        Args:
            transform (str): The name of the torchvision transform.
            **kwargs: Additional keyword arguments for the registration.
        """
        from atria_core.utilities.strings import _convert_to_snake_case

        from atria_transforms.core.torchvision import TorchvisionTransform

        self.register_modules(
            module_paths=TorchvisionTransform,
            module_names="image/" + _convert_to_snake_case(transform),
            transform=transform,
            **kwargs,
        )
