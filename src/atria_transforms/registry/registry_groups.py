from atria_registry import RegistryGroup


class DataTransformRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing data transformations.

    This class provides additional methods for registering and managing data
    transformations within the registry system.
    """

    def register(self, name: str, **kwargs):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """
        from atria_core.utilities.auto_config import auto_config

        def decorator(decorated_class):
            if hasattr(decorated_class, "_REGISTRY_CONFIGS"):
                configs = decorated_class._REGISTRY_CONFIGS
                assert isinstance(configs, dict), (
                    f"Expected _REGISTRY_CONFIGS on {decorated_class.__name__} to be a dict, "
                    f"but got {type(configs).__name__} instead."
                )
                assert configs, (
                    f"{decorated_class.__name__} must provide at least one configuration in _REGISTRY_CONFIGS."
                )
                for key, config in configs.items():
                    assert isinstance(config, dict), (
                        f"Configuration {config} must be a dict."
                    )
                    module_name = name
                    self.register_modules(
                        module_paths=decorated_class,
                        module_names=module_name + "/" + key,
                        **config,
                        **kwargs,
                    )
                return auto_config()(decorated_class)
            else:
                module_name = name
                self.register_modules(
                    module_paths=decorated_class, module_names=module_name, **kwargs
                )
                return auto_config()(decorated_class)

        return decorator

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
