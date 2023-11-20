<div align=center>
<img src="img/deepLink_logo.png">
</div>

# 国产硬件接入DIPU技术参考

## 1. 介绍
DIPU 是由一组抽象设备 Runtime 接口，一组框架能力相关的运行时基类/接口，一个针对 DIOPI 标准算子的适配层共同组成的拓展包，目的在于方便国产硬件接入并支持训练。本文档主要说明接入流程，提供标准化的技术接入参考实现。
## 2. 准备工作
### 2.1 环境准备
在使用国产硬件接入DIPU之前，我们需要先准备一个自己编译的Pytorch2.0（纯 CPU 版本即可），并确保自己的Pytorch2.0处于可用状态。这里需要确定使用的gcc、cmake、python3等基础库的版本尽可能匹配，同时确保这个环境能够编译硬件算子库。

以下步骤供参考：


#### 2.1.1 准备Python及gcc等

``` bash
# 准备python，如 3.8 版本
conda create --prefix=dipu python=3.8
conda activate /home/$USER/env/dipu

# 安装gcc，推荐 7.5
wget http://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz
tar -zxvf gcc-7.5.0.tar.gz
cd gcc-7.5.0
./contrib/download_prerequisites
# --prefix 请根据自己的需求修改
./configure --disable-multilib --enable-languages=c,c++ --prefix=/home/$USER/env/dipu/gcc
make -j20
make install
# 环境生效
cd /home/$USER/env/dipu/gcc/bin
export PATH=`pwd`:$PATH
```

#### 2.1.2 安装Pytorch
``` note
使用gcc 7.5编译pytorch
pytorch 2.0 推荐使用commitid：c263bd43e8e8502d4726643bc6fd046f0130ac0e
```

``` bash
cd /home/$USER/code
git clone git@github.com:pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
git checkout c263bd43e8e8502d4726643bc6fd046f0130ac0e
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
BUILD_BINARY=0 USE_PRECOMPILED_HEADERS=1 BUILD_TEST=0 USE_CUDA=0 python setup.py develop
```

### 2.1.3 编译DIPU

``` bash
cd /home/$USER/code
git clone git@github.com:DeepLink-org/dipu.git
cd dipu
git submodule update --init --recursive

# 修改 template_build _sh 中 PYTORCH_DIR、PYTHON_INCLUDE_DIR
# 示例
# PYTORCH_DIR="/home/$USER/code/pytorch"

# DIPU_DEVICE设置成厂商在dipu的设备名，即 https://github.com/DeepLink-org/dipu/blob/main/CMakeLists.txt 中的DEVICE_CAMB、DEVICE_ASCEND对应的字符串
# 示例
# export DIPU_DEVICE=camb
pip install .

```
### 2.1.4 验证DIPU
``` bash
export DIOPI_ROOT=/home/$USER/code/dipu/third_party/DIOPI/impl/lib
export DIPU_ROOT=/home/$USER/code/dipu/torch_dipu
export LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LIBRARY_PATH; 
export LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH

sh ./tests/python/run_tests.sh
```

## 3. 算子库

### 3.1 算子库接入（请参考DIOPI第三方芯片算子库）

在接入DIPU之前，我们的硬件应该提供一个已经实现的算子库，并已经按照 DIOPI的PROTO 声明进行了对应函数的实现，接入 DIOPI的IMPL。通过DIOPI的IMPL，我们在之前编译DIPU时会默认为对应设备编译出``libdiopi_impl.so``作为算子库文件
- 细节可参考 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)
- 需要注意的是，在我们进行一致性测试（diopi_test）时，会在编译时开启``DTEST=ON``，在我们接入DIPU时，编译的算子库应该关闭测试选项，即在cmake阶段使用``DTEST=OFF``
- 下面是一个 DIOPI的IMPL 中的算子接入样例
```c++
__global__ void softmaxKernel(const float* in, float* out, int64_t outer_dim, int64_t inner_dim, int64_t axis_dim) {
    for (int64_t k = threadIdx.x; k < inner_dim; k += blockDim.x) {
        const float *cur_in = in + blockIdx.x * axis_dim * inner_dim + k;
        float *cur_out = out + blockIdx.x * axis_dim * inner_dim + k;

        float max_val = cur_in[0];
        for (int64_t j = 0; j < axis_dim; ++j) {
            if (cur_in[j * inner_dim] > max_val) {
                max_val = cur_in[j * inner_dim];
            }
        }

        float exp_sum = (float)0.0;
        for (int64_t j = 0; j < axis_dim; ++j) {
            float exp_val = expf(cur_in[j * inner_dim] - max_val);
            cur_out[j * inner_dim] = exp_val;
            exp_sum += exp_val;
        }

        const float r_exp_sum = (float) 1.0 / exp_sum;
        for (int64_t j = 0; j < axis_dim; ++j) {
            cur_out[j * inner_dim] *= r_exp_sum;
        }
    }
}

DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim)
{
    auto stream  = impl::tang::getStream(ctx);
    auto trInput = impl::tang::makeTensor(input);
    auto trOut   = impl::tang::makeTensor(out);

    diopiSize_t inShape = trInput.shape();
    int64_t dim_count = inShape.len;
    int64_t real_axis = dim + (dim < 0 ? dim_count : 0);
    int64_t dim_shape = inShape.data[real_axis];
    int64_t inner_dim = 1;
    int64_t outer_dim = 1;
    for (int64_t i = 0; i < real_axis; ++i) {
        outer_dim *= inShape.data[i];
    }
    for (int64_t i = real_axis + 1; i < dim_count; ++i) {
        inner_dim *= inShape.data[i];
    }
    int32_t block_dimx = 512;
    int32_t grid_dimx = outer_dim;
    dim3 grid = grid_dimx;
    dim3 block = block_dimx;

    const void* inData = trInput.data();
    void* outData = trOut.data();

    void* args[] = {&inData, &outData, &outer_dim, &inner_dim, &dim_shape};
    DIOPI_CALLDROPLET(launchKernel(softmaxKernel, grid, block, args, 0, stream))

    return diopiSuccess;
}
```

### 3.2 算子库拓展功能
#### 3.2.1 算子fallback
Fallback 给定算子
```shell
$ export DIPU_FORCE_FALLBACK_OPS_LIST=add.out,conv2d
$ python -c "import torch_dipu"
```

Fallback scalar版本的重载函数， tensor版本的重载函数类似
```shell
$ export DIPU_FORCE_FALLBACK_OPS_LIST='.*.Scalar'
$ python -c "import torch_dipu"
```

Fallback 所有设备算子
```shell
$ export DIPU_FORCE_FALLBACK_OPS_LIST='.*'
$ python -c "import torch_dipu"
```

#### 3.2.2 算子精度自动对比功能介绍
由于该功能默认不开启，使用该功能时需要打开该功能并重新编译DIPU。
如在寒武纪设备上，可将`dipu/torch_dipu/csrc_dipu/CMakeLists.txt`中的`autocompare`修改为`True`
```
add_custom_command(
  OUTPUT "${DIPU_AUTOGENED_KERNELS_CPP}"
  COMMAND
    python "${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT}" --config
    "${DIPU_AUTOGEN_DIOPI_WRAPPER_CONFIG}" --out "${DIPU_AUTOGENED_KERNELS_CPP}"
    --use_diopi_adapter False --autocompare True --print_func_call_info True
    --print_op_arg True --fun_config_dict
    '{\"current_device\": \"${UsedVendor}\"}'
  DEPENDS ${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT}
          ${DIPU_AUTOGEN_DIOPI_WRAPPER_CONFIG})
```
以上方法是对所有算子开启自动精度对比，如果只需要对特定算子做精度对比，也可只给需要的算子做精度对比。
只需要在相关的配置文件（如`dipu/scripts/autogen_diopi_wrapper/diopi_functions.yaml`）给相应的算子添加`autocompare: True`即可

```shell
$ unset  DIPU_FORCE_FALLBACK_OPS_LIST # 主要是确保要比较的算子没有强制fallback到cpu,可选
$ python
>>> import torch
>>> import torch_dipu
>>> x = torch.randn(4,5,5).cuda()
>>> y = x + x

dipu_add_out_autocompare
autocompare:    add.out out:
        numel:100, sizes:[4, 5, 5], stride:[25, 5, 1], is_view:0, TensorOptions(dtype=float, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x9d1e480
        numel:100, sizes:[4, 5, 5], stride:[25, 5, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7efef9a00200
        allclose
autocompare:    add.out self: allclose
autocompare:    add.out other: allclose
>>> z = x + 3

dipu_add_out_autocompare
autocompare:    add.out out:
        numel:100, sizes:[4, 5, 5], stride:[25, 5, 1], is_view:0, TensorOptions(dtype=float, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x8f72ed00
        numel:100, sizes:[4, 5, 5], stride:[25, 5, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7efef9a00400
        allclose
autocompare:    add.out self: allclose
autocompare:    add.out other: allclose
>>>
```
可以看到，CPU计算结果与设备计算结果`allclose`，也能看到CPU和设备计算结果的`shape`，`dtype`等信息。特别的，需要注意以下几个问题：
1. `dipu/scripts/autogen_diopi_wrapper/diopi_functions.yaml`中配置了`autograd:True`的算子（`cross_entropy_loss`，`conv2d`，`dropout`，`dropout_`，`linear`）暂不支持*backward*的精度自动对比。如模型精度对不齐，可根据需要先将这几个算子fallback到CPU来确定问题。
2. 随机数生成相关的算子（`dipu/scripts/autogen_diopi_wrapper/diopi_functions.yaml`中配置了`autocompare:False`）没有做`autocompare`，因为结果总是 `not_allclose`。
3. 对输入做检查是确保算子输入不被意外修改。

#### 3.2.3 抓取算子参数
该功能需要打开`autogen`的`print_op_arg`和`print_func_call_info`选项，在模型调试和测试时遇到问题时可方便的拿到算子输入情况。不需要打印时也可关掉。

```shell
>>> import torch
>>> import torch_dipu
>>> import os
diopi dyload init
>>> x = torch.randn(3,4).cuda()
>>> os.environ['DIPU_DUMP_OP_ARGS']='1' # 只打印调用的底层算子名以及相关的diopi函数
>>> y = x + x
[dipu_add_out:349]:add.out  diopiAdd


>>> os.environ['DIPU_DUMP_OP_ARGS']='2'  # 打印调用的底层算子名，相关的diopi函数，算子参数
>>> y = x + 3
[dipu_add_out:349]:add.out  diopiAdd
[dipu_add_scalar_out:248]:add.Scalar_out  diopiAddScalar
        add.Scalar_out: self:numel:12, sizes:[3, 4], stride:[4, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7ff8c8c00000
        add.Scalar_out: other:3
        add.Scalar_out: alpha:1
        add.Scalar_out: out:numel:12, sizes:[3, 4], stride:[4, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7ff8c8c00400


>>> os.environ['DIPU_DUMP_OP_ARGS']='3' # 打印调用的底层算子名，相关的diopi函数，算子参数， tensor的值
>>> y = x * 3
[dipu_mul_out:815]:mul.out  diopiMul
[dipu_mul_scalar_out:753]:mul.Scalar_out  diopiMulScalar
        mul.Scalar_out: self:numel:12, sizes:[3, 4], stride:[4, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7ff8c8c00000
-0.4226 -0.4211 -1.5073  1.1861
-1.0474 -2.6718  0.4150  0.9834
 0.4800 -1.5484 -0.5011  0.2218
[ PrivateUse1FloatType{3,4} ]
        mul.Scalar_out: other:3
        mul.Scalar_out: out:numel:12, sizes:[3, 4], stride:[4, 1], is_view:0, TensorOptions(dtype=float, device=privateuseone:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)), data_ptr:0x7ff8c8c00200
 2.5774  2.5789  1.4927  4.1861
 1.9526  0.3282  3.4150  3.9834
 3.4800  1.4516  2.4989  3.2218
[ PrivateUse1FloatType{3,4} ]


>>> os.environ['DIPU_DUMP_OP_ARGS']='0'  # 不打印任何额外信息
>>> y = x / 3
>>>

```

## 4. 新硬件Runtime接入实现
### 4.1 接入流程示意图
 ![结构图](https://deeplink.readthedocs.io/zh_CN/latest/_images/SOP_01.png)
### 4.2 核心代码添加
- 在``dipu/torch_dipu/csrc_dipu/runtime/device/basedef.h``中定义了DIPU支持的硬件类型，我们需要在`VendorDeviceType`枚举类中添加 DROPLET 的硬件后端，并在这个文件中的`VendorTypeToStr`函数里添加新硬件支持。后续这个文件中可能有更多的函数会涉及到硬件类型，按需添加即可
- `dipu/torch_dipu/csrc_dipu/vendor`文件夹中存有各个硬件后端的*runtime*接入代码，我们需要根据`dipu/torch_dipu/csrc_dipu/runtime/device/deviceapis.h`中的声明，创建`deviceimpl.cpp`去根据硬件自己底层的*runtime*接口实现对应的函数。下面是`deviceapis.h`中的`createStream`函数的在国产硬件上的实现样例：

``` c++
void createStream(deviceStream_t* stream, bool prior) {
    if (prior) {
        DIPU_CALLDROPLET(::tangStreamCreateWithPriority(stream, tangStreamDefault, -1))
    } else {
        DIPU_CALLDROPLET(::tangStreamCreate(stream))
    }
}
```
- 如果有多机多卡训练的需求，需要根据`dipu/torch_dipu/csrc_dipu/runtime/device/diclapis.h`中的声明，创建`communiatorimpl.cpp`去根据硬件自己底层的*runtime*接口实现对应的函数
- DIPU在`dipu/torch_dipu/csrc_dipu/runtime/core/DIPUGeneratorImpl.h`中声明了`DIPUGeneratorImpl`这一个基本，如果我们的硬件实现了自己的`generator`基础函数，可以在这基础上实现自己的`DeviceGeneratorImpl`，并实现基础的`generator`相关函数。国产硬件暂无这方面的实现

### 4.3 增加编译脚本
- 在`dipu/CMakeList.txt`中，加入新硬件的控制代码。可以参考CUDA、CAMB等其他硬件，加入DROPLET选项，让打开`USE_DROPLET`，并使得`UsedVendor`变为DROPLET，同时添加该设备默认的DIOPI构建目标`DIOPI_IMPL_OPT`，并修改对应DIOPI构建目标的CMakeLists.txt文件，DIOPI的CMakeLists.txt修改细节可参考 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)。如果不准备在构建DIPU时同时构建DIOPI，可以将`DIOPI_IMPL_OPT`设置为`""`，
参考的示例代码如下
```
list(APPEND DEVICE_DROPLET "DROPLET" "droplet")
......
elseif (${DEVICE} IN_LIST DEVICE_DROPLET)
  set(USE_DROPLET ON)
  set(UsedVendor droplet)
  set(DIOPI_IMPL_OPT "droplet")
......
```

- 在`dipu/torch_dipu/csrc_dipu/vendor`中我们需要编写`CMakeList`，给出`VENDOR_INCLUDE_DIRS`、`VENDOR_LIB_DIRS`、`DIPU_VENDOR_LIB`、`VENDOR_FILES`这几个硬件后端自己的头文件、库文件和runtime接入源代码，来让上层根据这些变量进行编译
- 对应上述CMAKE的修改，我们应该修改我们的环境变量，将DIPU_DEVICE设置为`droplet`。

### 4.4 编译与测试
- 根据DIPU的编译介绍，我们在编译了DIPU之后，需要注意将`LIBRARY_PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`都设置好避免后续使用出现问题
- `dipu/tests`文件夹中有许多基础功能的测试，建议首先尝试测试`python -u dipu/tests/python/unittests/test_add.py`，该文件测试跑通基本意味着我们的设备*runtime*接入没有问题了
- 编译脚本参考[2.1.3](#213-编译dipu)，测试脚本可以参考[2.1.4](#214-验证dipu)

