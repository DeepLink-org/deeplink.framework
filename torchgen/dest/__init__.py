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

from .native_functions import (
    compute_native_function_declaration as compute_native_function_declaration,
)
from .register_dispatch_key import (
    gen_registration_headers as gen_registration_headers,
    gen_registration_helpers as gen_registration_helpers,
    RegisterDispatchKey as RegisterDispatchKey,
)
from .diopi_kernels import (
    gen_diopi_kernel as gen_diopi_kernel
)
