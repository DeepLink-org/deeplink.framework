import pytest
from model.utils import parse_bool_arg


def pytest_addoption(parser):
    parser.addoption("--backend", type=str, default=None)
    parser.addoption("--dynamic", type=parse_bool_arg, default=False)

@pytest.fixture(scope="session")
def backend(request):
    return request.config.getoption("--backend")

@pytest.fixture(scope="session")
def dynamic(request):
    return request.config.getoption("--dynamic")
