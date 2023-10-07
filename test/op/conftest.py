import pytest
from op.utils import parse_bool_arg


def pytest_addoption(parser):
    parser.addoption("--backend", type=str, default=None, required=False)
    parser.addoption("--need_dynamic", type=parse_bool_arg, default=False, required=False)
