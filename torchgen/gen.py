# Copyright (c) 2023, pjlab.org.cn
# Copyright (c) 2023, Meta CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from collections import defaultdict, namedtuple
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    Set,
    Union,
)

import yaml

from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    DispatchKey,
    FunctionSchema,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    SchemaKind,
)

from torchgen.utils import (
    concatMap,
    context,
    YamlLoader,
)

T = TypeVar("T")

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy dispatcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE = {}

def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> Set[str]:
    assert isinstance(es, list)
    rs: Set[str] = set()
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        tags = e.get("tag")
        with context(lambda: f"in {loc}:\n  {tags}"):
            e_i = e.copy()
            name = e_i.pop("tag")
            desc = e_i.pop("desc", "")
            # ensure that each tag has a non-empty description
            assert desc != ""
            rs.add(name)
    return rs

@functools.lru_cache(maxsize=None)
def parse_tags_yaml(path: str) -> Set[str]:
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path, "r") as f:
            es = yaml.load(f, Loader=LineLoader)
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])

def parse_native_yaml(
    path: str, 
    tags_yaml_path: str, 
    custom_path: str,
    ignore_keys: Optional[Set[DispatchKey]] = None,
) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)
        with open(path, "r") as f:
            es = yaml.load(f, Loader=LineLoader)
        assert isinstance(es, list)
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            loc = Location(path, e["__line__"])
            funcs = e.get("func")
            with context(lambda: f"in {loc}:\n  {funcs}"):
                func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
                rs.append(func)
                BackendIndex.grow_index(bs, m)

        # Filter the custom native yaml file, and extract the functions we defined.
        from io import StringIO
        f_str = StringIO()
        with open(custom_path, "r") as f:
            for line in f:
                if line.split(":")[0] in ["backend", "cpp_namespace", "extra_headers",
                                          "supported", "autograd", "custom", "custom_autograd"]:
                    continue
                if ":" not in line:
                    continue
                f_str.write(line)

        f_str.seek(0)
        custom_es = yaml.load(f_str, Loader=LineLoader)
        if custom_es != None:
            for e in custom_es:
                loc = Location(path, e["__line__"])
                funcs = e.get("func")
                with context(lambda: f"in {loc}:\n  {funcs}"):
                    func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
                    rs.append(func)
                    BackendIndex.grow_index(bs, m)

        error_check_native_functions(rs)
        # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
        indices: Dict[DispatchKey, BackendIndex] = defaultdict(
            lambda: BackendIndex(
                dispatch_key=DispatchKey.Undefined,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                # I'm actually not sure about this; undefined could be hit on
                # empty TensorList, hypothetically that could have sizes in it
                index={},
            )
        )
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                # Only cuda-like devices in tree require device guards
                device_guard=False,
                index=v,
            )
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = ParsedYaml(rs, indices)

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]

# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    base_func_map: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. "
                f"Consider adding 'structured=True' to the delegated operator"
            )
        # See Note [resize_ in Functionalization]
        # resize_() is technically an inplace view op (and therefore needs the tag),
        # but it would be overkill to add a true "view" variant of resize.
        # Instead, resize_() gets special treatment in functionalization,
        # and we have a resize() op that is non-aliasing + functional.
        if "inplace_view" in f.tags and str(f.func.name) != "resize_":
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, (
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming "
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            )
            out_of_place_base_name = BaseOperatorName(
                base_name.base, False, base_name.dunder_method
            )
            assert len(base_func_map[out_of_place_base_name]) > 0, (
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding "
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "
            )

def get_grouped_native_functions(
        native_functions: Sequence[NativeFunction]
) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    def flatten_pre_group(
            d: Dict[SchemaKind, NativeFunction]
        ) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            # Invariant: any NativeFunctions that are code-generated
            # should have been grouped into NativeFunctionsGroup objects
            assert not any("generated" in f.tags for f in d.values())
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions: Dict[
        FunctionSchema, Dict[SchemaKind, NativeFunction]
    ] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    
    return list(
        concatMap(flatten_pre_group, list(pre_grouped_native_functions.values()))
    )
