import acl
import os
import numpy as np
import time

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


# the range for dynamic shape
max_range = 32


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


buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }

def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))
        
total_compute_time = 0

def zero_total_compute_time():
    global total_compute_time
    total_compute_time = 0

def increase_compute_time(t):
    global total_compute_time
    total_compute_time += t
    
def get_total_compute_time():
    global total_compute_time
    return total_compute_time

class AscendExecutor(object):
    def __init__(self, device_id, dims, model_path) -> None:
        self.device_id = device_id          # int
        self.model_path = model_path        # str
        self.model_id = None                # pointer
        self.context = None                 # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None              # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.input_dims = dims

        self.init_resource()

    def release_resource(self):
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

    def init_resource(self):
        print("init resource stage:")
        # load model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)

        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            if des == "in" and self.input_dims is not None and i in self.input_dims.keys():
                tot_size = 1
                for elem in self.input_dims[i]:
                    tot_size *= elem
                dtype = acl.mdl.get_input_data_type(self.model_desc, i)
                np_dtype = get_np_dtype(dtype)
                temp_buffer_size = tot_size * np.dtype(np_dtype).itemsize

            if temp_buffer_size == 0:
                if self.input_dims is not None and des == "out":
                    dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
                    check_ret("acl.mdl.get_output_dims", ret)
                    dims = dims["dims"]
                    dims = [max_range if dim == -1 else dim for dim in dims]
                    temp_buffer_size = 1
                    for d in dims:
                        temp_buffer_size *= d
                    dtype = acl.mdl.get_output_data_type(self.model_desc, i)
                    np_dtype = get_np_dtype(dtype)
                    temp_buffer_size *= np.dtype(np_dtype).itemsize
                    temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                                 ACL_MEM_MALLOC_HUGE_FIRST)
                else:
                    temp_buffer, ret = acl.rt.malloc(1,
                                                 ACL_MEM_MALLOC_HUGE_FIRST)
                check_ret("acl.rt.malloc", ret)
            else:
                temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
                check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data

        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for i, item in enumerate(self.output_data):
                if self.input_dims is not None:
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
                else:
                    dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
                    check_ret("acl.mdl.get_cur_output_dims", ret)
                    out_dim = dims["dims"]
                    tot_size = item["size"]

                temp, ret = acl.rt.malloc_host(tot_size)
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": tot_size, "buffer": temp, "dims": out_dim})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                ptr = dataset[i]
                if item["size"] == 0:
                    continue
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    dataset[i]["size"],
                                    item["buffer"],
                                    dataset[i]["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

        return dataset

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        #print("data interaction from host to device")
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        #print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        #print("data interaction from device to host")
        res = []
        # copy device to host
        res = self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        #print("data interaction from device to host success")
        result = self.get_result(res)
        self._destroy_databuffer()
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return result

    def run(self, images):
        self._data_from_host_to_device(images)
        self.forward()
        return self._data_from_device_to_host()

    def forward(self):
        #print('execute stage:')
        if self.input_dims is not None:
            for key, value in self.input_dims.items():
                dtype = acl.mdl.get_input_data_type(self.model_desc, key)
                tensorDesc = acl.create_tensor_desc(0, value, dtype)
                dataset, ret = acl.mdl.set_dataset_tensor_desc(self.load_input_dataset,
                                                tensorDesc, key)
                check_ret("acl.mdl.set_dataset_tensor_desc", ret)
                assert(dataset == self.load_input_dataset)
        start = time.time()
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        end = time.time()
        increase_compute_time(end - start)
        check_ret("acl.mdl.execute", ret)
        # self._destroy_databuffer()
        # print('execute time: ', end - start)
        # print('execute stage success')

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

    def get_result(self, output_data):
        result = []
        
        for i, temp in enumerate(output_data):
            out_dim = temp["dims"]
            dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            np_dtype = get_np_dtype(dtype)

            ptr = temp["buffer"]
            bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
            data = np.frombuffer(bytes_data, dtype=np_dtype).reshape(tuple(out_dim))
            result.append(data)
        return result


class AscendModel():
    def __init__(self, device_id, model_path) -> None:
        self.device_id = device_id          # int
        self.model_path = model_path        # str

    def run(self, images, dims=None):
        exe = AscendExecutor(self.device_id, dims, self.model_path)
        result = exe.run(images)
        exe.release_resource()
        return result


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "build", "demo_add_add.om")

    exe = AscendExecutor(0, model_path)
    
    # make input data
    input_data = np.random.randn(1, 1, 28, 28)

    exe.run([input_data])

    print("*****run finish******")
    exe.release_resource()
