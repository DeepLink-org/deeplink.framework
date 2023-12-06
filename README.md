<!-- markdownlint-disable-next-line MD041 MD033 -->
<div align=center>
<!-- markdownlint-disable-next-line MD033 -->
<img src="https://deeplink.readthedocs.io/zh-cn/latest/_static/image/logo.png" alt="DeepLink Logo">
</div>

# DeepLink.Framework

Deeplink.framework 是 DeepLink 推出的介于 AI 训练框架和硬件语言之间的训练系统，同时支持 Eager 模式和 Graph 模式两种链路。Framework 仓库主要由 DIPU (Device Independent Process Unit) 和 DICP (Device Independent Compile Protocol) 两部分组成，framework 中对 Eager 模式的支持，主要由 DIPU 完成，framework 中对 Graph 模式的支持主要由 DICP 完成。

## 仓库结构

### DIPU

DIPU (Device Independent Process Unit) 是由一组抽象设备 runtime 接口，一组框架能力相关的运行时基类/接口，一个针对 DIOPI 标准算子的适配层共同组成的拓展包。用来在训练框架 PyTorch 上接入 DIOPI 算子库，实现 Eager 模式的推理和训练。其能够在编译时，决定抽象设备被影射的方式；并使用统一的运行时，减少在多硬件上适配训练框架的成本。DIPU 即可以基于统一的设备运行时来屏蔽厂商的实际设备；也可以基于统一的框架相关的运行时基类，由厂商自行实现特有的运行时逻辑。

### DICP

标准编译协议 (Device-Independent Compile Protocol, DICP) 定义了统一的计算描述（中间表示），通过计算图获取深度学习模型中的计算任务表达为上述中间表示，然后通过计算图优化技术自动生成人工智能芯片设备代码，从而提高研发效率和计算的执行性能。中间表示是介于源语言和目标语言之间的程序表示，能够极大程度地提高编译流程的可拓展性，同时也能降低优化流程对前端和后端的破坏。多层次中间表示包含从应用到芯片端的多种表示层次，不同层次旨在解决不同尺度的问题。

## 相关索引

* [DIPU 介绍](./dipu/README.md)
* [DIPU 文档](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/Introduction.html)
* [DICP 介绍](./dicp/readme.md)
* [DICP 文档](https://deeplink.readthedocs.io/zh-cn/latest/doc/DICP/introduction.html)
