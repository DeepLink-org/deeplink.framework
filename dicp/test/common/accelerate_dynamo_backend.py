import enum
from importlib import import_module
from typing import List
from contextlib import contextmanager


# Usage:
# with extend_dynamo_plugin_backend(["topsgraph", "ascendgraph"]):
#     model = torch.compile(model, backend="topsgraph")
@contextmanager
def extend_dynamo_plugin_backend(backend_list: List[str]):
    dynamo_plugin_module_str = "accelerate.utils.dataclasses"
    enum_class_to_extend_str = "DynamoBackend"

    dynamo_plugin_module = import_module(dynamo_plugin_module_str)
    origin_dynamo_backend_class = getattr(dynamo_plugin_module, enum_class_to_extend_str)
    dynamo_backend_name_value_dict = {item.name: item.value for item in origin_dynamo_backend_class}
    dynamo_backend_name_value_dict.update({backend.upper(): backend.upper() for backend in backend_list})
    extended_dynamo_backend_class = enum.Enum(enum_class_to_extend_str, names=dynamo_backend_name_value_dict)
    setattr(dynamo_plugin_module, enum_class_to_extend_str, extended_dynamo_backend_class)

    yield

    setattr(dynamo_plugin_module, enum_class_to_extend_str, origin_dynamo_backend_class)
