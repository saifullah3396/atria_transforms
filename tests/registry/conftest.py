import pytest
from atria_registry import RegistryGroup


@pytest.fixture(scope="class", autouse=True)
def test_registry_group():
    return RegistryGroup(name="mock_group", default_provider="atria_transforms")
