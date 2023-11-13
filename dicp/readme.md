<div align=center>
<img src="https://deeplink.readthedocs.io/zh-cn/latest/_static/image/logo.png">
</div>

# DICP

标准编译协议（Device-Independent Compile Protocol,DICP）定义了统一的计算描述（中间表示），通过计算图获取深度学习模型中的计算任务表达为上述中间表示，然后通过计算图优化技术自动生成人工智能芯片设备代码，从而提高研发效率和计算的执行性能。中间表示是介于源语言和目标语言之间的程序表示，能够极大程度地提高编译流程的可拓展性，同时也能降低优化流程对前端和后端的破坏。多层次中间表示包含从应用到芯片端的多种表示层次，不同层次旨在解决不同尺度的问题。

DICP主要的核心功能如下：
1. **通过接入编译路线带来性能优势，在大模型场景最大限度释放芯片能力**
2. **作为训练框架与国产硬件芯片之间的通用桥梁，支持多种前后端，带来使用易用性**
3. **提供易用、高效的一站式编译适配流程，灵活支持国产硬件图编译器的特性，提高芯片适配效率**

下图描述了DICP在编译链路中的位置：

<div align=center>
<img src="https://deeplink.readthedocs.io/zh-cn/latest/_static/image/DICP/dicp_flow.png">
<p>*DICP在编译链路中的位置</p>

</div>

1. 训练框架通过图获取模块将用户的模型代码转换成统一的中间表达。此处的中间表达完全与芯片无关。所以在之后的编译协议部分中，需要建立起与后端芯片的联系。这样才能高效的完成接入。
2. 编译协议完成了衔接框架与芯片编译器的工作，其中包含硬件相关的切图，统一中间表达与芯片所支持的算子之间的映射关系以及数据格式的转换模块。
3. 在编译协议吸收了芯片特点之后，由代码生成模块生成最终的代码，并通过芯片的编译器生成二进制可执行文件之后由框架调用。



## 基于DICP的国产硬件接入PyTorch2实践

<!-- ### DICP vs 纯Dynamo -->

基于上述DICP，国产硬件可快速接入Pytorch2的编译路线。此路线中的TorchDynamo组件，可使国产硬件在运行时的overhead大幅缩小。  
并且针对国产硬件实现了以下特性：
  - 灵活支持国产硬件图编译器的特性
  - 支持多种国产硬件数据格式
  - 支持动态shape

### 运行逻辑
DICP的运行逻辑如下图所示:
<!-- (**这张图有问题，需要讨论 by jinminxi**) -->

<div align=center>
<img src="https://deeplink.readthedocs.io/zh-cn/latest/_static/image/DICP/structure.png">
</div>

其中：
1. **算子映射**： 主要解决框架层算子与后端图编译器的算子之间的语义差别，包括1对1和1对多的转换。  
2. **Shape&Dtype推导**： 进行Shape&data_type的推导，补全整张静态图上的信息，便于之后在代码生成模块能生成代码。  
3. **子图改写**： 将多个小算子融合成为一个或多个适合图编译器的算子，配合后端图编译器将计算效率最大化。
4. **数据格式调整**： 是根据后端芯片与其图编译器的特性，针对特定的算子调整其输入输出的数据格式，使得最大程度的发挥芯片性能。

### 目录结构
* dicp/dynamo_bridge： 多后端通用的接入代码，包含了
  1. 接收从AOTAutograd下发而来的FX Graph
  2. 启动各个厂商的IR转换与优化
  3. 启动CodeGen以及JIT缓存的逻辑。
* dicp/vender: 主要包含了各个厂商IR的定义，AtenIR到厂商IR的转换，厂商IR上的优化以及最后的代码生成模块。
* test: 包含了model测试与op测试


### Demo

#### 安装DICP

```
cd /path_to_dicp
pip install .
```

#### 在华为910上执行llama7B前向推理
```
export DIPU_MOCK_CUDA = false
export DICP_TOPS_DIPU = True
export TEST_DIR = /path_to_dicp/test/
export LLAMA_MODEL_DIR=/path_to_llama_model
bash /path_to_dicp/test/model/run_test_model.sh llama ascendgraph false
```

#### 在燧原T20上执行resnet50训练
```
export DIPU_MOCK_CUDA = false
export DICP_TOPS_DIPU = True
export TEST_DIR = /path_to_dicp/test/
bash /path_to_dicp/test/model/run_test_model.sh resnet50 topsgraph false
```
