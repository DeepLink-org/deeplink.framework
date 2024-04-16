import atexit
import os
import time
import math

import ctypes
import gc


import acl
import numpy as np
import torch
import torch_dipu
from torch.profiler import record_function

dipu_device_str = torch_dipu.dipu.device.__diputype__

# error code
ACL_SUCCESS = 0
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
            self.work_size = int(3 * 1024 * 1024 * 1024)
            self.work_tensor = torch.empty(
                self.work_size, dtype=torch.bool, device=dipu_device_str)
            self.work_ptr = self.work_tensor.data_ptr()

    def release_memory(self):
        print("Release bufferPtr from MemoryPool.")
        self.work_tensor = None


class GraphManager:
    def __init__(self):
        device_id = torch_dipu.current_device()
        self._lib_path = os.environ.get(
            "DICP_ASCEND_GE_GRAPH_EXECUTOR", "/tmp/dicp_ascend/ge_graph.so")
        self.graph_manager = ctypes.CDLL(self._lib_path)
        self.graph_manager.init(device_id)
        atexit.register(self.release_graph)

    def release_graph(self):
        self.graph_manager.release()


zero_tensor = torch.empty(1, device=dipu_device_str)
graph_manager = None
memory_pool = MemoryPool()
graph_id = 0


def get_graph_manager():
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
    return graph_manager.graph_manager


class GEStaticGraphExecutor(object):
    def __init__(self, graph_id, device_id):
        self.device_id = device_id
        self.graph_id = graph_id

        # init
        self.const_mem_size = graph_manager.graph_manager.get_const_size(
            self.graph_id)
        self.workspace_mem_size = graph_manager.graph_manager.get_workspace_size(
            self.graph_id)

        # alloc memory
        self.const_tensor = torch.empty(
            self.const_mem_size, dtype=torch.bool, device='dipu')
        self.const_ptr = self.const_tensor.data_ptr()
        graph_manager.graph_manager.set_graph_memory(self.graph_id, ctypes.c_void_p(
            self.const_ptr), ctypes.c_void_p(memory_pool.work_ptr), self.const_mem_size, self.workspace_mem_size)

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
        dtypes = output_dtype_str.split(',')

        assert len(shapes) == len(dtypes)

        self.output_shapes = []
        self.output_dtypes = []
        self.output_datasize = []
        for item in shapes:
            elems = item.split(',')
            elems = [int(x) for x in elems]
            self.output_shapes.append(elems)
        for item in dtypes:
            elem = int(item)
            self.output_dtypes.append(elem)
        for i in range(len(shapes)):
            elem_size = math.prod(self.output_shapes[i])
            self.output_datasize.append(
                elem_size * acl.data_type_size(self.output_dtypes[i]))
        self.output_datasize_c = (
            ctypes.c_int64 * len(self.output_datasize))(*self.output_datasize)

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
        dtypes = input_dtype_str.split(',')

        assert len(shapes) == len(dtypes)

        self.input_shapes = []
        self.input_dtypes = []
        self.input_datasize = []

        for item in shapes:
            elems = item.split(',')
            elems = [int(x) for x in elems]
            self.input_shapes.append(elems)
        for item in dtypes:
            elem = int(item)
            self.input_dtypes.append(elem)
        for i in range(len(shapes)):
            elem_size = math.prod(self.input_shapes[i])
            self.input_datasize.append(
                elem_size * acl.data_type_size(self.input_dtypes[i]))
        self.input_datasize_c = (
            ctypes.c_int64 * len(self.input_datasize))(*self.input_datasize)

    @record_function(f'load_and_run_run')
    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None,
            allocated_output=None):
        inputs = [x.to(dipu_device_str) if isinstance(x, torch.Tensor)
                  and x.device.type != dipu_device_str else x for x in images]

        input_ptrs = [x.data_ptr() for x in inputs]
        input_ptrs_c = (ctypes.c_void_p * len(inputs))(*input_ptrs)
        output_ptrs = []
        output_tensors = []

        if allocated_output:
            allocated_output_tensor = {}
            for output_index, input_index in allocated_output.items():
                allocated_output_tensor[output_index] = inputs[input_index]

        for i, shape in enumerate(self.output_shapes):
            if allocated_output and i in allocated_output.keys():
                item = allocated_output_tensor[i]
            else:
                item = torch.empty(shape, dtype=get_tensor_dtype(
                    self.output_dtypes[i]), device=dipu_device_str)
            output_ptrs.append(item.data_ptr())
            output_tensors.append(item)

        output_ptrs_c = (ctypes.c_void_p * len(output_tensors))(*output_ptrs)
        current_stream = torch_dipu.current_stream(self.device_id).dipu_stream
        graph_manager.graph_manager.run(self.graph_id, current_stream, input_ptrs_c,
                                        output_ptrs_c, self.input_datasize_c, self.output_datasize_c)
        return output_tensors


class GEModel():
    def __init__(self, graph_id, device_id, is_static=True) -> None:
        atexit.register(self.cleanup)
        if is_static:
            self.exe = GEStaticGraphExecutor(graph_id, device_id)
        else:
            raise RuntimeError("current GEModel only support static graph!!")

    def run(self, images, dims=None, output_shape=None,
            out_stride=None, out_storage_offset=None, allocated_output=None):
        return self.exe.run(images, dims, output_shape, out_stride, out_storage_offset, allocated_output)

    def cleanup(self):
        if hasattr(self, 'exe'):
            del self.exe


if __name__ == '__main__':
    pass

