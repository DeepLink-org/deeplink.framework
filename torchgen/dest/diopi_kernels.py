# Copyright (c) 2023, pjlab.org.cn
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

import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from typing_extensions import Literal

import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    Expr,
    kernel_signature,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)

@dataclass(frozen=True)
class DiopiKernel:
    backend_index: BackendIndex