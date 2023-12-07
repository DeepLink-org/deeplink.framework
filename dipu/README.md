<div align=center>
<img src="https://deeplink.readthedocs.io/zh-cn/latest/_static/image/logo.png">
</div>

# DIPU

## 介绍
DIPU (Device Independent Process Unit) 是由 **一组抽象设备 Runtime 接口，一组框架能力相关的运行时基类/接口，一个针对 DIOPI 标准算子的适配层** 共同组成的拓展包。 用来在训练框架 PyTorch 上接入 DIOPI 算子库，实现 Eager 模式的推理和训练。其能够在编译时，决定抽象设备被影射的方式；并使用统一的运行时，减少在多硬件上适配训练框架的成本。DIPU 即可以基于统一的设备运行时来屏蔽厂商的实际设备；也可以基于统一的框架相关的运行时基类，由厂商自行实现特有的运行时逻辑。

虽然 PyTorch 定义了一套基础的运行时接口``c10``，可以基于这个接口直接抽象各个设备接口，但是``c10``首先是个直面框架层的接口，每个接入的设备都需要实现大量类似的逻辑来完成``c10``的实现，对于多设备的支持很不方便。DIPU 先把``c10``的运行时适配到 DIPU 自己的运行时，把通用的逻辑抽取出来，可以让厂商仅实现必要的设备接口即可工作。


## 代码结构说明

<img src="https://deeplink.readthedocs.io/zh_CN/latest/_images/structure1.png">

DIPU 结构上分为 Python 和 CPP 两部分。

### CPP层
#### 1. Runtime (``csrc/dipu/runtime``):
  *Runtime* 主要有以下几个部分：

##### 1）*Core & Distributed*：
PyTorch 把一些基本的设备层接口放到了一个叫 ``c10`` 的目录下，不同的设备接入者需要实现该接口来接入 PyTorch。
详见[参考文档](http://blog.ezyang.com/2019/05/pytorch-internals/) 里对于``c10`` 的介绍。

DIPU 的这一部分主要就是对 PyTorch 的``c10`` 和``c10d``相关接口的实现，把设备无关的部分抽象出一组运行时基类。目前包含 ``DIPUAllocator``，``DIPUGenerator``，``DIPUStream/Event/Guard``，``ProcessGroupDICL`` 等。这些类会把设备相关的请求代理到 *device* 部分定义的一组设备接口。另外用户也可以继承上述基类，实现并注册自己的子类，实现设备特化的某些行为( 这个能力的支持目前尚待完善)。

##### 2）*Device*:

包含 ``deviceapis.h`` 和 ``diclapis.h`` 两个接口文件。主要是设备 ``memory/stream/event/communcation`` 相关的接口函数（这部分接口后续有考虑挪到 DIOPI 中，成为 DIOPI 的 *Device* 接口，见上图）。

#### 2. Aten (``csrc/dipu/aten``):
  这块的能力主要依赖于 PyTorch 本身的 注册新 *backend* 的能力，DIPU 并没有在 PyTorch 的 源码里新增 *Backend Key*，而是使用了已有的 "``PrivateUse1``" 这个 key。

  主要功能是把 ATen 的 *backend* 层算子适配到 DIOPI 标准算子库。ATen 的 *backend* 层算子基本可以认为是 ATen IR，它定义的算子集合 和 DIOPI 算子库是高度一致的。因此这块目前只有较少的算子转换/组合逻辑，大多数时候都是直接从 ATen 算子映射到 DIOPI 算子。 这一部分以后可以支持设备特化的算子注册（规划中）。

  另外，并不是所有的算子实现都代理到 DIOPI 。对于 *view* 型算子和内存分配型算子，DIPU 目前是自行实现的。

#### 3. DiopiRT (``csrc/dipu/diopirt``):
   用于实现 DIOPI 要求的 *Runtime* ，具体参考 [DIOPI项目](https://github.com/DeepLink-org/DIOPI)。

#### 4. Binding to Python (``csrc/dipu/binding``):
   主要用于导出 DIPU *Runime* 接口到 Python，并定义一些在CPP层做 ``monkey-patch`` 的 PyTorch 原生函数（这部分后面会谨慎新增）。

#### 5. Vendor (``csrc/dipu/vendor``):
   硬件设备相关的接口 和 编译选项要实现在这里。

   一般的，除了要实现上面 *Device* 部分要求的接口函数外，*Vendor* 还需要实现一个特殊的 ``vendorapi.h``，在这里导出设备 ``device/stream/event/comm`` 相关的数据结构定义。未来计划在设备层允许*Vendor* 注册特化的 *Runtime* 子类，或者实现子类的构建器/工厂方法接口，实现设备特化的 *Runtime* 行为。

### Python层
  1. DIPU 设备层接口 (``torch_dipu/dipu``):

      包含CPP层的 *Runtime* 接口对应的Python层。这部分会导出部分函数给用户侧，导出的函数类比 PyTorch 的 ``torch/cuda`` 部分。

  2. DIPU 采用 ``monkey-patch`` 的方式模拟了部分 Pytorch tensor 接口，让他们可以处理 DIPU 特殊的参数，这部分的设计还在变化中。


  3. DIPU 拥有一定的模拟 CUDA 接口的能力。简单来说就是在 Python 层 用前面 DIPU 设备层的接口来替换 ``torch.cuda`` 的同名接口。 


  后面另有规划 DIPU 的配置化接口等能力，可以为不同的 *Vendor* 输入不同配置。以配置驱动的方式来指导 *Runtime* 和 DIOPI 算子适配流程的构建。

## 相关功能介绍:
### Dispatch 机制与 DIOPI 算子库
  PyTorch 的算子注册和分派有很多步骤，详见[参考文档](
  https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough)。

  DIPU CPP层适配的 ATen 算子对应的是分派过程中最底层（*backend*层） 的算子或者 *composite* 层里等效为 *backend* 的算子。

  这里面有一定的灵活性，以``Linear``算子为例，在 PyTorch 的 ``cpu/cuda`` 设备上，它被实现为一个 ``composite`` 算子，实际的 *backend* 层算子是组合算子内部调用的 ``addmm`` 或者更底层的 ``mm``。 而在 DIPU (``privateuse1``) 设备中，目前是注册了 一个 ``Linear`` 算子 ( DIOPI 有这个算子 ) 来替代组合算子，所以分派会直接走到新的 *backend* 层算子 ``Linear`` ，而不会在调用原来的 ``addmm/mm``。但是如果对应设备的 DIOPI 的 IMPL 算子库 没有实现 ``diopiLinear`` 而是实现了 ``mm`` 算子，也是可以正常走通 ``Linear`` 的调用流程的。

### 无侵入式的 PyTorch 扩展包:
  DIPU 没有直接修改 PyTorch 的代码，而是使用 out-of-tree 的方式接入新设备，详见[参考文档](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)。
  
  PyTorch 要求 out-of-tree 的代码 必须定义一个私有的 ``Backend Key``，DIPU目前没有和 PyTorch 做官方的沟通， 因此 PyTorch 主干里没有``DIPU``这个设备，目前是暂时借用``PrivateUse1`` 这个 Key (后续考虑改为借用``XPU``设备 Key，因为这个 Key 在 PyTorch 主干代码中有更好的支持)。

  基于用户私有的 ``Backend Key`` 和 ``Dispatch Key``，PyTorch 会把算子调用请求分发到对应设备的算子实现。另外``c10`` 本身提供了一些注册能力，比如    ``C10_REGISTER_GUARD_IMPL``，可以让用户把私有设备的 *Runtime* 代码注册到框架中。

  但是 PyTorch 并不完全符合 `扩展开放，修改关闭` 的范式。很多能力不是基于`注册` 的方式来开放给扩展组件的，而是在代码里对不同的 ``Backend Key`` 做的 if-else 判断。 并且不同的组件对于 ``Backend Key`` 的支持程度也不同。 有些 Legacy 的逻辑只支持 CUDA & CPU，完全无法扩展; 还有一些仅支持固定的几个 ``Backend Key``。DIPU 目前的做法是 在 Python 层加一层代理，把用户的函数调用转换成底层可以支持的方式。这样的问题是会带来很多无谓的适配逻辑，但是鉴于 PyTorch 的现状，暂时先这样处理。后续也希望和 PyTorch 官方有所合作。


### 算子适配能力
  为了更好的接入 DIOPI 算子，DIPU 提供了一组 算子适配相关的辅助能力，比如灵活的算子 Fallback to CPU 的能力，算子精度自动对比的能力（对比 DIOPI 算子 和 PyTorch 原生的 CPU 算子），算子执行过程中打印算子参数的能力。基于这些能力，接入算子时可以更方便排查算子精度等问题。 相关能力的具体说明参见 [Quick Start 文档](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/quick_start.html)的 *算子库接入*。


## 质量保障体系
在每次代码合并之前，都会在各个设备上跑测试，测试全都跑通过才能合并。
我们的测试包括三部分：
1. Pytorch 测例。我们充分利用了 Pytorch 的测试框架的功能，能够充分利用 Pytorch 的测例。有些情况下，Pytorch 测例过于严苛，超过了设备的支持能力时，也可以在配置文件中跳过相关测例。可以为每个设备单独设置：算子精度阈值，支持的数据类型，要跳过的测例，要跑的测例等。
2. 简单开发的手工测例。这部分测例更注重算子能否跑通，对算子要求较低。
3. 模型测试。我们开发了``one_iter``精度对比工具，会先在精度正确性没问题的设备（如 CPU 和 CUDA ）上训练模型，保存每一层的算子输入、输出、权重、梯度数据，再在待测试设备上训练模型，逐层对比训练精度。

更多信息请参考：[dipu/tests](https://github.com/DeepLink-org/DIPU/tree/main/dipu/tests)。

## Learn More

* [Quick Start](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/quick_start.html)
* [Profiler](./torch_dipu/profiler/readme.md)
* [常见问题](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/FAQ.html)
* [开发者指南](https://github.com/DeepLink-org/DIPU/tree/main/dipu/Contributors.md)
