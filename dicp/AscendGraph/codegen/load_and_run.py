import acl
import os
import numpy as np
import torch
import time


# the range for dynamic shape
max_range = 128

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

def get_np_dtype(dtype):
    if dtype == ACL_FLOAT:
        return np.float32
    elif dtype == ACL_INT64:
        return np.int64
    elif dtype == ACL_INT32:
        return np.int32
    elif dtype == ACL_BOOL:
        return np.bool_
    elif dtype == ACL_DOUBLE:
        return np.float64
    elif dtype == ACL_COMPLEX64:
        return np.complex64
    elif dtype == ACL_FLOAT16:
        return np.float16
    raise RuntimeError("unsupported np dtype!")


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


class AscendExecutor(object):
    def __init__(self, device_id, dims, model_path) -> None:
        self.device_id = device_id          # int
        self.model_path = model_path        # str
        self.model_id = None                # pointer
        self.context = None                 # pointer
        self.model_desc = None              # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.num_inputs = 0
        self.num_outputs = 0
        self.input_size = []
        self.output_size = []
        self.output_dims = []
        self.output_dtypes = []
        self.output_data = []
        self.input_dims = dims

        self.init_resource()

    def release_resource(self):
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        
        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item)
            check_ret("acl.rt.free", ret)
            
    def load_model(self):
        config_handle = acl.mdl.create_config_handle()
        ret = acl.mdl.set_config_opt(config_handle, ACL_MDL_LOAD_TYPE_SIZET, 1)
        check_ret("set_config_opt", ret) 

        ret = acl.mdl.set_config_opt(config_handle, ACL_MDL_PATH_PTR, self.model_path)
        check_ret("set_config_opt", ret)

        ret = acl.mdl.set_config_opt(config_handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, 1)
        check_ret("set_config_opt", ret)

        self.model_id, ret = acl.mdl.load_with_config(config_handle)
        check_ret("acl.mdl.load_with_config", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

    def init_resource(self):
        print("init resource stage:")

        self.load_model()
        self.num_inputs = acl.mdl.get_num_inputs(self.model_desc)
        self.num_outputs = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(self.num_inputs):
            temp_buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            if self.input_dims is not None and i in self.input_dims.keys():
                tot_size = 1
                for elem in self.input_dims[i]:
                    tot_size *= elem
                dtype = acl.mdl.get_input_data_type(self.model_desc, i)
                np_dtype = get_np_dtype(dtype)
                temp_buffer_size = tot_size * np.dtype(np_dtype).itemsize
            self.input_size.append(temp_buffer_size)
        for i in range(self.num_outputs):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            check_ret("acl.mdl.get_output_dims", ret)
            dims = dims["dims"]    
            if temp_buffer_size == 0:
                if self.input_dims is not None:
                    dims = [max_range if dim == -1 else dim for dim in dims]   
                    temp_buffer_size = 1
                    for d in dims:
                        temp_buffer_size *= d
                    np_dtype = get_np_dtype(dtype)
                    temp_buffer_size *= np.dtype(np_dtype).itemsize
                else:
                    temp_buffer_size = 1
            self.output_dtypes.append(get_tensor_dtype(dtype))
            self.output_dims.append(dims)
            self.output_size.append(temp_buffer_size)

        print("init resource success")

    def _prepare_input(self, images):
        assert self.num_inputs == len(images)
        self.load_input_dataset = acl.mdl.create_dataset()
        zero_tensor = torch.randn(1).to('dipu')
        for i in range(self.num_inputs):
            if self.input_size[i] == 0:
                ptr = zero_tensor.data_ptr()
            else:
                ptr = images[i].data_ptr()
            data = acl.create_data_buffer(ptr, self.input_size[i])
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

        if self.input_dims is not None:
            for key, value in self.input_dims.items():
                dtype = acl.mdl.get_input_data_type(self.model_desc, key)
                format = acl.mdl.get_input_format(self.model_desc, key)
                tensorDesc = acl.create_tensor_desc(dtype, value, format)
                dataset, ret = acl.mdl.set_dataset_tensor_desc(self.load_input_dataset,
                                                tensorDesc, key)
                check_ret("acl.mdl.set_dataset_tensor_desc", ret)
                assert(dataset == self.load_input_dataset)

    def _prepare_output(self, output_tensor):
        self.load_output_dataset = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            item = torch.empty(self.output_dims[i], dtype=self.output_dtypes[i], device='dipu')
            output_tensor.append(item)
            data = acl.create_data_buffer(item.data_ptr(), self.output_size[i])
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _prepare_tmp_output(self):
        self.load_output_dataset = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            temp_buffer, ret = acl.rt.malloc(self.output_size[i],
                            ACL_MEM_MALLOC_HUGE_FIRST)            
            data = acl.create_data_buffer(temp_buffer, self.output_size[i])
            self.output_data.append(temp_buffer)
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _prepare_real_output(self, output_tensor):
        for i in range(self.num_outputs):
            out = acl.mdl.get_dataset_tensor_desc(self.load_output_dataset, i)
            tsize = acl.get_tensor_desc_num_dims(out)
            out_dim = []
            tot_size = 1
            for d in range(tsize):
                out_dim.append(acl.get_tensor_desc_dim(out, d))
                tot_size *= out_dim[-1]
            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            np_dtype = get_np_dtype(dtype)
            tot_size *= np.dtype(np_dtype).itemsize
            item = torch.empty(out_dim, dtype=self.output_dtypes[i], device='dipu')
            output_tensor.append(item)
            ret = acl.rt.memcpy(item.data_ptr(),
                                tot_size,
                                self.output_data[i],
                                tot_size,
                                ACL_MEMCPY_DEVICE_TO_DEVICE)
            check_ret("acl.rt.memcpy", ret)            

    def run(self, images):
        assert len(images) > 0
        input = list(map(lambda x: x.to('dipu'), images))
        self._prepare_input(input)
        output = []
        if self.input_dims is not None:
            self._prepare_tmp_output()
            self.forward()
            self._prepare_real_output(output)
        else:
            self._prepare_output(output)
            self.forward()
        self._destroy_databuffer()
        return output

    def forward(self):
        #start = time.time()
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        #print('forward2 time:', time.time() - start)
        check_ret("acl.mdl.execute", ret)

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)


# class AscendModel():
#     def __init__(self, device_id, model_path) -> None:
#         self.device_id = device_id          # int
#         self.model_path = model_path        # str

#     def run(self, images, dims=None):
#         exe = AscendExecutor(self.device_id, dims, self.model_path)
#         result = exe.run(images)
#         exe.release_resource()
#         return result
    
class AscendModel():
    def __init__(self, device_id, model_path) -> None:
        self.device_id = device_id          # int
        self.model_path = model_path        # str
        self.exe = AscendExecutor(self.device_id, None, self.model_path)

    def run(self, images, dims=None):
        return self.exe.run(images)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "build", "demo_add_add.om")

    exe = AscendExecutor(0, model_path)
    
    # make input data
    input_data = np.random.randn(1, 1, 28, 28)

    exe.run([input_data])

    print("*****run finish******")
    exe.release_resource()
