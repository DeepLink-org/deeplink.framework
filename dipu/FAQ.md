# 常见问题

## 国产硬件如何接入 DIPU？

请参考 [Quick Start](./QuickStart.md)

## 加载 DIPU 模块训练模型，出现 `loss` 异常如何排查？

根据项目组模型训练的经验，`loss` 异常通常是因为 `DIOPI` 算子实现有 `bug` 导致的。比较常见的一个 `bug` 是  `DIOPI` 算子实现未考虑不连续的 `tensor`，建议可以先从这个角度先梳理算子的实现。另外还可以通过以下步骤来定位

1. 模型用到的所有算子都用 `DIOPI` 一致性测试验证正确性(验证范围包含模型测例和算子测例)。如果未通过，定位修复，直至通过。
2. 跑 DIPU 测例 ，从 `DIPU` 角度验证算子实现的正确性。
3. 修改 `autogen` 配置，将 `autocompare` 和 `print_op_arg` 设置成 `True` ；此时 `autogen` 生成的代码将会自动比对 `DIPU` 算子执行结果和 `CPU` 执行结果，发现不匹配的情况，具体比对逻辑可以阅读生成的 `torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp`。如果日志中出现关键字 `not_close`， 则说明跑模型过程中发现了算子执行结果错误的情况，此时可以从日志中拿到算子输入参数信息，构造最小复现集，排查该算子问题。注：`autocompare` 功能并不能覆盖所有的算子，例如 `conv2d backward` 操作并不会做 `autocompare`。
4. 如果 `autocompare` 仍然没有发现问题，则可以通过设置环境变量 `DIPU_FORCE_FALLBACK_OPS_LIST` 将算子加入黑名单，此时该算子会 `fallback` 到 `CPU`。假设怀疑`conv2d`算子，可以将`conv2d` 加入黑名单 (`export DIPU_FORCE_FALLBACK_OPS_LIST=conv2d`)，此时 `conv2d` 算子将会 `fallback` 到 `CPU`，观察 `loss` 是否正常。如果 `loss` 现在恢复正常，则说明 `conv2d` 存在 `bug`；如果 `loss` 仍然异常，则可以增加更多的算子到黑名单，不断试验，得到有问题的算子。

## 使用 DIPU 出现显存泄露，如何定位？

`DIPU` 在几款芯片上都进行了测试，未发现显存泄露的问题；若厂商适配过程中出现显存泄露问题，可以重点排查DIOPI算子实现是否造成了内存泄露。

同时 DIPU 包含 [memory checker](./torch_dipu/csrc_dipu/runtime/core/MemChecker.cpp) 模块，能够协助排查内存泄露问题。当程序退出时，如果显存没被释放，会打印出分配显存的 `backtrace`；通过分析 `backtrace`，梳理变量的生命周期，即可定位显存泄露问题。

如若想开启 `memory checker` 功能，`export` 以下环境变量即可：

```bash
export DIPU_MEM_CHECK=1
export DIPU_MEM_CHECK_ENABLE_BACKTRACE=1
```

## 如果仍然无法找到问题

您可在项目中提交 issue，将您遇到的问题告诉我们。
