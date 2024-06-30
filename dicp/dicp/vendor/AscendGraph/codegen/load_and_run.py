import atexit
import ctypes
import os
import math

from dicp.vendor.AscendGraph.codegen.utils import get_ascend_dtype_num, get_ascend_format_num, get_torch_dtype

from ctypes import POINTER, c_longlong, c_size_t, c_void_p, c_int64, c_int

from pathlib import Path

import acl
import numpy as np
import torch
import torch_dipu
from torch.profiler import record_function

dipu_device_str = torch_dipu.dipu.device.__diputype__

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2
# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
# error code
ACL_SUCCESS = 0
# images format
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']
# data format
NPY_FLOAT32 = 11
ACL_DT_UNDEFINED = -1
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12
ACL_COMPLEX64 = 16

ACL_MDL_PRIORITY_INT32 = 0
ACL_MDL_LOAD_TYPE_SIZET = 1
ACL_MDL_PATH_PTR = 2
ACL_MDL_MEM_ADDR_PTR = 3
ACL_MDL_MEM_SIZET = 4
ACL_MDL_WEIGHT_ADDR_PTR = 5
ACL_MDL_WEIGHT_SIZET = 6
ACL_MDL_WORKSPACE_ADDR_PTR = 7
ACL_MDL_WORKSPACE_SIZET = 8
ACL_MDL_INPUTQ_NUM_SIZET = 9
ACL_MDL_INPUTQ_ADDR_PTR = 10
ACL_MDL_OUTPUTQ_NUM_SIZET = 11
ACL_MDL_OUTPUTQ_ADDR_PTR = 12
ACL_MDL_WORKSPACE_MEM_OPTIMIZE = 13

ACL_DDR_MEM = 0
ACL_HBM_MEM = 1
ACL_DDR_MEM_HUGE = 2
ACL_DDR_MEM_NORMAL = 3
ACL_HBM_MEM_HUGE = 4
ACL_HBM_MEM_NORMAL = 5
ACL_DDR_MEM_P2P_HUGE = 6
ACL_DDR_MEM_P2P_NORMAL = 7
ACL_HBM_MEM_P2P_HUGE = 8
ACL_HBM_MEM_P2P_NORMAL = 9


def get_tensor_dtype(dtype):
    if dtype == ACL_FLOAT:
        return torch.float32
    elif dtype == ACL_INT64:
        return torch.int64
    elif dtype == ACL_FLOAT16:
        return torch.float16
    elif dtype == ACL_INT32:
        return torch.int32
    elif dtype == ACL_BOOL:
        return torch.bool
    elif dtype == ACL_DOUBLE:
        return torch.float64
    elif dtype == ACL_COMPLEX64:
        return torch.complex64
    raise RuntimeError(f"can not convert acl dtype:{dtype} to torch dtype")


buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}


def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class MemoryPool:
    def __init__(self):
        self.work_ptr = None
        atexit.register(self.release_memory)
        self.init_work_weight_ptr()

    def init_work_weight_ptr(self):
        if self.work_ptr is None:
            free, _, ret = acl.rt.get_mem_info(ACL_HBM_MEM)
            check_ret("acl.rt.get_mem_info", ret)
            
            self.work_size = int(6 * 1024 * 1024 * 1024)
            self.work_tensor = torch.empty(
                self.work_size, dtype=torch.bool, device=dipu_device_str)
            self.work_ptr = self.work_tensor.data_ptr()

    def release_memory(self):
        print("Release bufferPtr from MemoryPool.")
        self.work_tensor = None


class GraphCompiler:
    def __init__(self):
        self._lib_path = os.environ.get(
            "DICP_ASCEND_GE_GRAPH_EXECUTOR", "/tmp/dicp_ascend/ge_graph.so")
        self.graph_compiler = ctypes.CDLL(self._lib_path)


class GraphManager:
    def __init__(self):
        device_id = torch_dipu.current_device()
        self._lib_path = os.environ.get(
            "DICP_ASCEND_GE_GRAPH_EXECUTOR", "/tmp/dicp_ascend/ge_graph.so")
        self.config_file = os.path.join(
            str(Path(__file__).resolve().parent), 'ge_init_config.json')
        self.graph_manager = ctypes.CDLL(self._lib_path)
        
        context, ret = acl.rt.get_context()
        check_ret("acl.rt.get_context", ret)
        self.graph_manager.init((c_void_p)(context), device_id, self.config_file.encode())
        atexit.register(self.release_graph)

    def release_graph(self):
        self.graph_manager.release()


zero_tensor = torch.empty(1, device=dipu_device_str)
graph_manager = None
graph_compiler = None
memory_pool = MemoryPool()
graph_id = 0


def get_graph_manager():
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
    return graph_manager.graph_manager


def get_graph_compiler():
    global graph_compiler
    if graph_compiler is None:
        graph_compiler = GraphCompiler()
    return graph_compiler.graph_compiler


class GEStaticGraphExecutor(object):
    def __init__(self, graph_id, device_id):
        self.device_id = device_id
        self.graph_id = graph_id
        print('### graph_id:', self.graph_id)

        # init
        self.const_mem_size = graph_manager.graph_manager.get_const_size(
            self.graph_id)
        self.feature_mem_size = graph_manager.graph_manager.get_feature_size(
            self.graph_id)

        # alloc memory
        self.const_tensor = torch.empty(
            self.const_mem_size, dtype=torch.bool, device='dipu')
        self.const_ptr = self.const_tensor.data_ptr()
        graph_manager.graph_manager.set_graph_memory(self.graph_id, c_void_p(
            self.const_ptr), c_void_p(memory_pool.work_ptr), c_size_t(self.const_mem_size), c_size_t(memory_pool.work_size))

        # prapare output info
        output_shape_buffer = ctypes.create_string_buffer(10000)
        output_dtype_buffer = ctypes.create_string_buffer(10000)
        graph_manager.graph_manager.get_output_shapes(
            self.graph_id, output_shape_buffer)
        graph_manager.graph_manager.get_output_dtypes(
            self.graph_id, output_dtype_buffer)

        output_shape_str = output_shape_buffer.value.decode('utf-8')
        output_dtype_str = output_dtype_buffer.value.decode('utf-8')
        shapes = output_shape_str.split(';')
        dtypes = output_dtype_str.split(';')

        assert len(shapes) == len(dtypes)

        self.output_shapes = []
        self.output_dtypes = []
        self.output_datasize = []
        for item in shapes:
            if item == '':
                self.output_shapes.append([])
                continue
            elems = item.split(',')
            elems = [int(x) for x in elems]
            self.output_shapes.append(elems)
        for item in dtypes:
            elem = int(item)
            self.output_dtypes.append(elem)
        for i in range(len(shapes)):
            elem_size = math.prod(self.output_shapes[i]) if len(
                self.output_shapes[i]) > 0 else 1
            self.output_datasize.append(
                elem_size * acl.data_type_size(self.output_dtypes[i]))
        self.output_datasize_c = (
            c_int64 * len(self.output_datasize))(*self.output_datasize)

        # prapare input info
        input_shape_buffer = ctypes.create_string_buffer(50000)
        input_dtype_buffer = ctypes.create_string_buffer(50000)
        graph_manager.graph_manager.get_input_shapes(
            self.graph_id, input_shape_buffer)
        graph_manager.graph_manager.get_input_dtypes(
            self.graph_id, input_dtype_buffer)

        input_shape_str = input_shape_buffer.value.decode('utf-8')
        input_dtype_str = input_dtype_buffer.value.decode('utf-8')
        shapes = input_shape_str.split(';')
        dtypes = input_dtype_str.split(';')

        assert len(shapes) == len(dtypes)

        self.input_shapes = []
        self.input_dtypes = []
        self.input_datasize = []

        for item in shapes:
            if item == '':
                self.input_shapes.append([])
                continue
            elems = item.split(',')
            elems = [int(x) for x in elems]
            self.input_shapes.append(elems)
        for item in dtypes:
            elem = int(item)
            self.input_dtypes.append(elem)
        for i in range(len(shapes)):
            elem_size = math.prod(self.input_shapes[i]) if len(
                self.input_shapes[i]) > 0 else 1
            self.input_datasize.append(
                elem_size * acl.data_type_size(self.input_dtypes[i]))
        self.input_datasize_c = (
            c_int64 * len(self.input_datasize))(*self.input_datasize)

    @record_function('load_and_run_run')
    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None,
            allocated_output=None):

        def get_data_ptr(data):
            if data.device.dtype != dipu_device_str:
                data = data.to(dipu_device_str)
            return data.data_ptr()

        inputs = [x.to(dipu_device_str) if isinstance(x, torch.Tensor)
                  and x.device.type != dipu_device_str else x for x in images]
        input_ptrs = [x.data_ptr() for x in inputs]

        input_ptrs_c = (c_void_p * len(inputs))(*input_ptrs)
        output_ptrs = []
        output_tensors = []

        if allocated_output:
            allocated_output_tensor = {}
            for output_index, input_index in allocated_output.items():
                allocated_output_tensor[output_index] = inputs[input_index]
            for i, shape in enumerate(self.output_shapes):
                if i in allocated_output.keys():
                    item = allocated_output_tensor[i]
                else:
                    item = torch.empty(shape, dtype=get_torch_dtype(
                        self.output_dtypes[i]), device=dipu_device_str)
                output_ptrs.append(item.data_ptr())
                output_tensors.append(item)
        else:
            for i, shape in enumerate(self.output_shapes):
                item = torch.empty(shape, dtype=get_torch_dtype(
                    self.output_dtypes[i]), device=dipu_device_str)
                output_ptrs.append(item.data_ptr())
                output_tensors.append(item)

        output_ptrs_c = (c_void_p * len(output_tensors))(*output_ptrs)
        context, ret = acl.rt.get_context()
        current_stream, ret = acl.rt.create_stream()
        graph_manager.graph_manager.run((c_void_p)(context), self.graph_id, current_stream, input_ptrs_c,
                                        output_ptrs_c, self.input_datasize_c, self.output_datasize_c)
        ret = acl.rt.synchronize_stream(current_stream)
        ret = acl.rt.destroy_stream(current_stream)
        check_ret("acl.rt.synchronize_stream", ret)
        return output_tensors


class GEDynamicGraphExecutor(object):
    def __init__(self, graph_id, device_id, input_nodes, output_nodes):
        self.device_id = device_id
        self.graph_id = graph_id
        self.is_first_run = True
        self.input_dtypes = []
        self.input_formats = []
        self.output_dtypes = []
        self.output_formats = []
        self.input_ascend_dtype_nums = []
        self.output_ascend_dtype_nums = []
        self.input_args_size = None
        self.input_args_dtypes_array = None
        self.input_args_formats_array = None
        self.output_args_size = None
        self.output_args_dtypes_array = None
        self.output_args_formats_array = None
        print('### graph_id:', self.graph_id)

        # init
        self.fixed_feature_mem_size = graph_manager.graph_manager.get_fixed_feature_size(
            self.graph_id)

        # alloc memory
        self.fixed_feature_tensor = torch.empty(
            self.fixed_feature_mem_size, dtype=torch.bool, device='dipu')
        self.fixed_feature_ptr = self.fixed_feature_tensor.data_ptr()
        graph_manager.graph_manager.set_fixed_feature_graph_memory(self.graph_id, ctypes.c_void_p(
            self.fixed_feature_ptr), self.fixed_feature_mem_size)

        # get input/output dtypes and formats
        self.input_args_size = len(input_nodes)
        for input in input_nodes:
            dtype = get_ascend_dtype_num(input['data_type'])
            format = get_ascend_format_num(input['format'])
            self.input_formats.append(format)
            self.input_dtypes.append(get_torch_dtype(dtype))
            self.input_ascend_dtype_nums.append(dtype)
        self.input_args_dtypes_array = (
            c_int * self.input_args_size)(*self.input_ascend_dtype_nums)
        self.input_args_formats_array = (
            c_int * self.input_args_size)(*self.input_formats)

        self.output_args_size = len(output_nodes)
        for output in output_nodes:
            dtype = get_ascend_dtype_num(output['data_type'])
            format = get_ascend_format_num(output['format'])
            self.output_ascend_dtype_nums.append(dtype)
            self.output_dtypes.append(get_torch_dtype(dtype))
            self.output_formats.append(format)
        self.output_args_dtypes_array = (
            c_int * self.output_args_size)(*self.output_ascend_dtype_nums)
        self.output_args_formats_array = (
            c_int * self.output_args_size)(*self.output_formats)

    @record_function('load_and_run_run')
    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None,
            allocated_output=None):
        assert len(images) > 0
        inputs = [x.to(dipu_device_str) if isinstance(x, torch.Tensor)
                  and x.device.type != dipu_device_str else x for x in images]
        input_ptrs = [x.data_ptr() for x in inputs]

        input_ptrs_c = (c_void_p * len(inputs))(*input_ptrs)
        output_ptrs = []
        output_tensors = []

        allocated_output_tensor = None
        if allocated_output:
            allocated_output_tensor = {}
            for output_index, input_index in allocated_output.items():
                allocated_output_tensor[output_index] = inputs[input_index]

        # assemble inputs/outputs
        cur_input_shapes = []
        cur_per_input_shape_size = []
        cur_output_shapes = []
        cur_per_output_shape_size = []
        input_datasize = []
        output_datasize = []
        for index, i in enumerate(inputs):
            shape = list(i.shape)
            shape = shape if shape != [] else [1]
            shape_data_size = math.prod(shape)
            input_datasize.append(
                shape_data_size * acl.data_type_size(self.input_ascend_dtype_nums[index]))
            cur_input_shapes.append((c_longlong * len(shape))(*shape))
            cur_per_input_shape_size.append(len(shape))
        input_args_shapes_array_size = (
            c_size_t * len(inputs))(*cur_per_input_shape_size)
        input_args_shapes_array = (
            POINTER(c_longlong) * len(inputs))(*cur_input_shapes)

        for index, shape in enumerate(output_shape):
            shape = shape if shape != [] else [1]
            shape_data_size = math.prod(shape)
            dtype = self.output_ascend_dtype_nums[index]
            output_datasize.append(
                shape_data_size * acl.data_type_size(dtype))
            cur_output_shapes.append((c_longlong * len(shape))(*shape))
            cur_per_output_shape_size.append(len(shape))
        output_args_shapes_array_size = (
            c_size_t * len(output_shape))(*cur_per_output_shape_size)
        output_args_shapes_array = (
            POINTER(c_longlong) * len(output_shape))(*cur_output_shapes)

        if self.is_first_run:
            graph_manager.graph_manager.assemble_inputs(self.graph_id, input_args_shapes_array, input_args_shapes_array_size,
                                                        self.input_args_size, self.input_args_dtypes_array, self.input_args_formats_array)
            graph_manager.graph_manager.assemble_outputs(self.graph_id, output_args_shapes_array, output_args_shapes_array_size,
                                                         self.output_args_size, self.output_args_dtypes_array, self.output_args_formats_array)
            self.is_first_run = False
        else:
            graph_manager.graph_manager.update_inputs(
                self.graph_id, input_args_shapes_array, input_args_shapes_array_size, self.input_args_size)
            graph_manager.graph_manager.update_outputs(
                self.graph_id, input_args_shapes_array, input_args_shapes_array_size, self.output_args_size)

        input_datasize_c = (
            c_int64 * len(input_datasize))(*input_datasize)
        output_datasize_c = (
            c_int64 * len(output_datasize))(*output_datasize)

        if allocated_output:
            allocated_output_tensor = {}
            for output_index, input_index in allocated_output.items():
                allocated_output_tensor[output_index] = inputs[input_index]
            for i, shape in enumerate(output_shape):
                if i in allocated_output.keys():
                    item = allocated_output_tensor[i]
                else:
                    item = torch.empty(
                        shape, dtype=self.output_dtypes[i], device=dipu_device_str)
                output_ptrs.append(item.data_ptr())
                output_tensors.append(item)
        else:
            for i, shape in enumerate(output_shape):
                item = torch.empty(
                    shape, dtype=self.output_dtypes[i], device=dipu_device_str)
                output_ptrs.append(item.data_ptr())
                output_tensors.append(item)

        output_ptrs_c = (c_void_p * len(output_tensors))(*output_ptrs)
        context, ret = acl.rt.get_context()
        current_stream, ret = acl.rt.create_stream()
        graph_manager.graph_manager.run((c_void_p)(context),self.graph_id, current_stream, input_ptrs_c,
                                        output_ptrs_c, input_datasize_c, output_datasize_c)
        ret = acl.rt.synchronize_stream(current_stream)
        ret = acl.rt.destroy_stream(current_stream)
        check_ret("acl.rt.synchronize_stream", ret)
        return output_tensors


class GEModel():
    def __init__(self, graph_id, device_id, is_static=True, input_nodes=None, output_nodes=None) -> None:
        atexit.register(self.cleanup)
        if is_static:
            self.exe = GEStaticGraphExecutor(graph_id, device_id)
        else:
            self.exe = GEDynamicGraphExecutor(
                graph_id, device_id, input_nodes, output_nodes)

    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None, allocated_output=None):
        return self.exe.run(images, dims, output_shape, out_stride, out_storage_offset, allocated_output)

    def cleanup(self):
        if hasattr(self, 'exe'):
            del self.exe



class AscendExecutor(object):
    def __init__(self, device_id, model_path) -> None:
        self.device_id = device_id          # int
        self.model_path = model_path        # str
        self.model_id = None                # pointer
        self.context = None                 # pointer
        self.model_desc = None              # pointer when using
        self.num_inputs = 0
        self.num_outputs = 0
        self.input_size = []
        self.output_size = []
        self.output_dims = []
        self.output_dtypes = []
        self.output_data = []
        self.input_shape = []
        self.input_dataset = acl.mdl.create_dataset()
        self.input_data_buffers = []
        self.output_dataset = acl.mdl.create_dataset()
        self.output_data_buffers = []
        self.weight_ptr = None

        self.init_resource()

    def __del__(self):
        self.release_resource()

    def release_resource(self):
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)
            self.model_id = None
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.weight_ptr is not None:
            ret = acl.rt.free(self.weight_ptr)
            check_ret("acl.rt.free", ret)
            self.weight_ptr = None

    def load_model(self):
        work_size, weight_size, ret = acl.mdl.query_size(self.model_path)
        check_ret("acl.mdl.query_size", ret)
        if work_size == 0:
            work_size = memory_pool.work_size
        elif work_size > memory_pool.work_size:
            free, _, ret = acl.rt.get_mem_info(ACL_HBM_MEM)
            check_ret("acl.rt.get_mem_info", ret)
            # If free < work_size, means that memory is insufficient.
            # Just ignore and continue, it may be work.
            if free > work_size:
                memory_pool.work_size = work_size
                memory_pool.release_memory()
                print("Adjust memory pool allocation.")
                memory_pool.work_ptr, ret = acl.rt.malloc(work_size,
                                                          ACL_MEM_MALLOC_HUGE_FIRST)
                check_ret("acl.rt.malloc", ret)

        self.weight_ptr, ret = acl.rt.malloc(weight_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
        check_ret("acl.rt.malloc", ret)
        config_handle = acl.mdl.create_config_handle()
        ret = acl.mdl.set_config_opt(config_handle, ACL_MDL_LOAD_TYPE_SIZET, 2)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_PATH_PTR, self.model_path)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_WEIGHT_ADDR_PTR, self.weight_ptr)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_WEIGHT_SIZET, weight_size)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_WORKSPACE_ADDR_PTR, memory_pool.work_ptr)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_WORKSPACE_SIZET, memory_pool.work_size)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(
            config_handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, 1)
        check_ret("set_config_opt", ret)

        self.model_id, ret = acl.mdl.load_with_config(config_handle)
        check_ret("acl.mdl.load_with_config", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

    def init_resource(self):
        self.load_model()
        self.num_inputs = acl.mdl.get_num_inputs(self.model_desc)
        self.num_outputs = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(self.num_inputs):
            temp_buffer_size = acl.mdl.get_input_size_by_index(
                self.model_desc, i)
            self.input_size.append(temp_buffer_size)
            input_dims, ret = acl.mdl.get_input_dims(self.model_desc, i)
            check_ret("acl.mdl.get_input_dims", ret)
            self.input_shape.append(input_dims)
            data_buf = acl.create_data_buffer(0, 1)
            self.input_data_buffers.append(data_buf)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buf)
            check_ret("acl.add_dataset_buffer", ret)

        for i in range(self.num_outputs):
            temp_buffer_size = acl.mdl.get_output_size_by_index(
                self.model_desc, i)
            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            check_ret("acl.mdl.get_output_dims", ret)
            self.output_dtypes.append(get_torch_dtype(dtype))
            self.output_dims.append(dims["dims"])
            self.output_size.append(temp_buffer_size)
            data_buf = acl.create_data_buffer(0, 1)
            self.output_data_buffers.append(data_buf)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buf)
            check_ret("acl.add_dataset_buffer", ret)

    @record_function('load_and_run_prepare_input')
    def _prepare_input(self, images, dims):
        assert self.num_inputs == len(images)
        for i in range(self.num_inputs):
            buffer_size = self.input_size[i]
            if dims is not None and i in dims.keys():
                tot_size = 1
                for elem in dims[i]:
                    tot_size *= elem
                dtype = acl.mdl.get_input_data_type(self.model_desc, i)
                buffer_size = tot_size * acl.data_type_size(dtype)

            if buffer_size == 0:
                buffer_size = 1
                ptr = zero_tensor.data_ptr()
            else:
                ptr = images[i].data_ptr()

            ret = acl.update_data_buffer(
                self.input_data_buffers[i], ptr, buffer_size)
            check_ret("acl.update_data_buffer", ret)

            if dims is not None and i in dims.keys():
                dtype = acl.mdl.get_input_data_type(self.model_desc, i)
                format = acl.mdl.get_input_format(self.model_desc, i)
                tensorDesc = acl.create_tensor_desc(dtype, dims[i], format)
                dataset, ret = acl.mdl.set_dataset_tensor_desc(self.input_dataset,
                                                               tensorDesc, i)
                check_ret("acl.mdl.set_dataset_tensor_desc", ret)
                assert (dataset == self.input_dataset)

    @record_function('load_and_run_prepare_output')
    def _prepare_output(self, output_tensor, output_shape, out_stride, out_storage_offset, allocated_output):
        for i in range(self.num_outputs):
            if allocated_output and i in allocated_output.keys():
                item = allocated_output[i]
            else:
                item = torch.empty(
                    self.output_dims[i], dtype=self.output_dtypes[i], device=dipu_device_str)
            # TODO! add case judgement for stride info
            # item = item.as_strided(
            #     self.output_dims[i], out_stride[i], out_storage_offset[i])
            output_tensor.append(item)
            ret = acl.update_data_buffer(
                self.output_data_buffers[i], item.data_ptr(), self.output_size[i])
            check_ret("acl.update_data_buffer", ret)

    @record_function('load_and_run_prepare_dynamic_output')
    def _prepare_dynamic_output(self, output_tensor, output_shape, out_stride, out_storage_offset, allocated_output):
        for i in range(self.num_outputs):
            tot_size = 1
            for elem in output_shape[i]:
                tot_size *= elem
            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            tot_size *= acl.data_type_size(dtype)
            self.output_dims[i] = output_shape[i]
            self.output_size[i] = tot_size
            if allocated_output and i in allocated_output.keys():
                item = allocated_output[i]
            else:
                item = torch.empty(
                    self.output_dims[i], dtype=self.output_dtypes[i], device=dipu_device_str)
            # TODO! add case judgement for stride info
            # item = item.as_strided(
            #     self.output_dims[i], out_stride[i], out_storage_offset[i])

            output_tensor.append(item)
            ret = acl.update_data_buffer(
                self.output_data_buffers[i], item.data_ptr(), self.output_size[i])
            check_ret("acl.update_data_buffer", ret)

    @record_function('load_and_run_run')
    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None,
            allocated_output=None):
        assert len(images) > 0
        input = [x.to(dipu_device_str) if isinstance(x, torch.Tensor)
                 and x.device.type != dipu_device_str else x for x in images]
        allocated_output_tensor = None
        if allocated_output:
            allocated_output_tensor = {}
            for output_index, input_index in allocated_output.items():
                allocated_output_tensor[output_index] = input[input_index]

        self._prepare_input(input, dims)
        output = []
        if output_shape:
            self._prepare_dynamic_output(
                output, output_shape, out_stride, out_storage_offset, allocated_output_tensor)
        else:
            self._prepare_output(
                output, output_shape, out_stride, out_storage_offset, allocated_output_tensor)
        self.forward()
        self._destroy_databuffer()
        return output

    @record_function('load_and_run_forward')
    def forward(self):
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_dataset)
        check_ret("acl.mdl.execute", ret)

    def _destroy_databuffer(self):
        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item)
            check_ret("acl.rt.free", ret)


class AscendModel():
    def __init__(self, device_id, model_path) -> None:
        atexit.register(self.cleanup)
        self.exe = AscendExecutor(device_id, model_path)

    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None, allocated_output=None):
        return self.exe.run(images, dims, output_shape, out_stride, out_storage_offset, allocated_output)

    def cleanup(self):
        if hasattr(self, 'exe'):
            del self.exe


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "build", "demo_add_add.om")

    exe = AscendExecutor(0, model_path)

    # make input data
    input_data = np.random.randn(1, 1, 28, 28)

    exe.run([input_data])

    print("*****run finish******")
    exe.release_resource()
