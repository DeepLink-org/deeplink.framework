import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", type=str, default=None)
    parser.addoption("--need_dynamic", type=bool, default=False)
