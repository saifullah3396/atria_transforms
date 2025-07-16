from atria_registry.module_registry import ModuleRegistry
from atria_registry.registry_group import RegistryGroup
from utilities import MockClass2

from atria_transforms import DATA_TRANSFORM


def test_registry_imports():
    assert DATA_TRANSFORM == ModuleRegistry().DATA_TRANSFORM and isinstance(
        DATA_TRANSFORM, RegistryGroup
    )


def test_register_all_modules():
    ModuleRegistry()["DATA_TRANSFORM"].register("test_module2")(MockClass2)
    ModuleRegistry().register_all_modules()

    def assert_module_in_registry(group, module_name, registry_group):
        module_found = False
        for x in ModuleRegistry()[registry_group].registered_modules:
            if x == (group, module_name):
                module_found = True
                break
        if not module_found:
            raise AssertionError(
                f"Module {module_name} in group {group} not found in registry group {registry_group}"
                f"Registered modules: {ModuleRegistry()[registry_group].registered_modules.keys()}"
            )

    assert_module_in_registry("data_transform", "test_module2", "DATA_TRANSFORM")
