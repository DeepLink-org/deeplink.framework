# Metrcis

本文将主要介绍本目录下的 metrcis 组件的理念、构成及使用方式。

## Metrcis 概念

本目录的组件用于生成如下所示的 Prometheus 风格的 metrcis 数值：

```text
allocator_event_count{device="0",event="nullptr",iteration="1",method="allocate",type="bfc"} 0
```

相关基础概念，请参考 Prometheus 相关资料：

- [Prometheus: OVERVIEW](https://prometheus.io/docs/introduction/overview/#what-are-metrics)
- [Prometheus: METRIC TYPES](https://prometheus.io/docs/concepts/metric_types/)。

## 设计

本目录下的 Metrcis 组件主要包含这些类型：

1. Metrcis 值的生命周期管理：
   - `Collector`
   - `LabeledValue`
   - `detail::collector`
   - `detail::group`
   - `detail::shared_value`
2. Metrcis 值类型：
   - `counter`
   - `gauge`
   - `histogram`
3. Metrcis 标签类型：
   - `detail::label`
   - `detail::labelset`
4. 其它未分类的帮助类型：
   - `ExportedGroup`
   - `detail::value_access`
   - `detail::value_operation`

> 备注：
>
> - 本模块中，小写命名的类型，以及使用 `detail` 包裹的类型，均为内部类型，通常来说使用者不应该直接使用它们。
> - 使用者有时候需要自定义 `labelset`，此时不要直接使用 `detail::labelset` 模板类型，而应该使用 `Collector::labelset`。`Collector::labelset` 帮助定义了所需的 `String` 参数类型。

### 生命周期管理

Metrcis value 的生命周期管理是树形的，从顶层的 `Collector` 到底层的 `detail::shared_value`，它们的关系如下所示：

```text
                                      ┌───────────────────┐                           
                                      │ Collector         │                           
                                      ├───────────────────┤                           
                                      │ detail::collector │                           
                                      └─────────▲─────────┘                           
                            1:1                 │                                     
                          ┌─────────────────────┼───────────────────┐                 
                          │                     │                   │                 
                 ┌────────┴─────────┐ ┌─────────┴────────┐ ┌────────┴─────────┐       
                 │ detail::group<A> │ │ detail::group<B> │ │ detail::group<C> │       
                 └────────▲─────────┘ └──────────────────┘ └────────▲─────────┘       
               n:1        │                                         │                 
             ┌────────────┘                                         │                 
             │                                                      │                 
┌────────────┴────────────┐  ┌─────────────────────────┐  ┌─────────┴───────────────┐ 
│ detail::shared_value<A> │  │ ......                  │  │ detail::shared_value<C> │ 
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘ 
```

上图没有出现的 `LabeledValue` 则是 `detail::shared_value` 的一个代理，它为使用者提供不同 metrics value 的操作接口，同时利用引用计数帮助使用者控制 value 类型的生命周期。通常来说，使用者只需要使用 `Collector` 构造对应的 `Labeled*` 类型即可。例如：

```cpp
// Create a LabeledIntegerCounter value.
auto allocate_nullptr_count = metrics::default_collector()
    .make_integer_counter("allocator_event_count", "")
    .with({{"method", "allocate"}, {"event", "nullptr"}});
```

在本方案中，`LabeledXXX` 类型的生命周期由创建它的关联代码片段控制，而默认 `Collector` 则具有全局生命周期，并间接被 Python 层调用（Pybind11）。设计方案借助引用计数，保证了关联的 `Labeled*` 和 `Collector` 实例的析构不会导致对方失效。

### 标签与数值

每个 metrics 数值都可以携带一组形如 `a=b` 的标签，这些标签将数值划分至不同维度。

例如前面的例子：

```text
allocator_event_count{device="0",event="nullptr",iteration="1",method="allocate",type="bfc"} 0
```

表示名为 `allocator_event_count` 的 counter 类型数值（单调递增），其中标签组为

- `device=0`
- `event=nullptr`
- `iteration=1`
- `method=allocate`
- `type=bfc`

的那一个数值的值为 0。

同名 metrics 数值的类型必须一致，允许有不同标签，且不同标签的数值是没有直接关联的。对不同标签进行聚合即可得到不同维度的该数值。

> 备注：切忌滥用标签，可能造成[维度爆炸](https://prometheus.io/docs/practices/instrumentation/#do-not-overuse-labels)，导致数值无法被聚合。

关于标签的使用，除了前面提到的直接使用初始化列表构造，还可以参考这个例子：

```cpp
auto a = metrics::Collector::labelset({{"key_1", "value_1"}}); // Create with string.
auto b = a({{"key_2", "value_2"}}); // Create the new label b with a and key_2.
b -= {"key_1"}; // Remove labels by key.
b += a; // Merge a into b.
```

## 使用方式

### Metrics 数值的定义及使用

可以直接使用默认的 `metrics::default_collector` 构造需要的数值。这种使用方式适用于简单直观，不涉及额外处理的情况：

```cpp
// Create and register metrics value.
auto static throughput_sum = metrics::default_collector()
    .make_integer_counter("collected_bytes", "some helper message");

// Update metrics values.
throughput_sum.add(nbytes);
```

值得注意的是，在相同的 `Collector` 对象中，同一组名字及标签将指向相同一个 metrics value。这就意味着在不同地方使用相同名字、相同标签参数构造的 metrics value 其实是同一个 value 的引用。

> 创建新的 metrics value 之前，`Collector` 对象会首先检查是否已经存在对应的值，如果不存在则创建，否则引用计数增加。由于查找过程（字符串匹配）存在开销，因此不建议在热路径反复执行查找操作，这里推荐创建一次，反复引用的做法。

如果需要创建一组关联的、跨函数使用的 metrics 数值，则推荐创建一个 wrapper 类型帮助处理这种情况。详细可以参考 [allocator_metrics.h](..\runtime\core\allocator\allocator_metrics.h) 文件。

### 数据上报

为了更好的与 Python 层交互，采集到的 metrics value 将会通过 Pybind11 导出。众所周知 Pybind11 无法直接导出自定义类型，这里需要将其转化为 Python 类型，或者转化为常见 STL 类型由 Pybind11 自动转化。

本模块的 `ExportedGroup` 用于进行数据交换。借助 `ExportedGroup::from_collector` 方法，遍历 collector 关联的 metrics value 并构造出对应的 `ExportedGroup` 数组。

通常来说我们并不需要关注 `ExportedGroup`，在 Python 层我们能直接获取它对应的导出类型：

```python
import torch_dipu
output = torch_dipu._C.metrics()
```

这份代码中 `output` 是数组类型，其中的元素可以用如下 Protocol 表示：

```python
class MetricsGroup(Protocol):
    name: str
    type: str
    help: str
    #                       labels,       value |        histogram ( thresholds,   buckets, sum)
    values: list[(list[(str, str)], int | float | tuple[list[int] | list[float], list[int], int])]
```

数组中的 `MetricsGroup` 则相当于前文 `detail::group` 的导出结构，它指代一组相同名字（及类型）metrics 数值。下面这个例子将介绍如何解析它并上传至 InflexDB：

```python
import random

from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from math import inf
from typing import Protocol


class MetricsGroup(Protocol):
    name: str
    type: str
    help: str
    #                       labels,       value |        histogram ( thresholds,   buckets, sum)
    values: list[(list[(str, str)], int | float | tuple[list[int] | list[float], list[int], int])]


@dataclass
class MetricsExporter:
    labels: dict[str, str | int | float | None] = field(default_factory = lambda: {})
    values: dict[str, str | int | float | None] = field(default_factory = lambda: {})

    # group -> name, suffix, labels, value
    def parse_metrics_groups(self, groups: Iterable[MetricsGroup]) -> Generator[tuple[str, str, list[(str, str)], int | float]]:
        for group in groups:
            match group.type:
                case "counter" | "gauge":
                    for (labels, value) in group.values:
                        yield (group.name, "", self._patch_labels(labels), value)

                case "histogram":
                    for (labels, (thresholds, buckets, summation)) in group.values:
                        accumulation = 0
                        for (threshold, bucket) in zip(thresholds + [inf], buckets):
                            accumulation += bucket
                            yield (group.name, "_bucket", self._patch_labels(labels, [("le", threshold)]), accumulation)
                        yield (group.name, "_sum", self._patch_labels(labels), summation)
                        yield (group.name, "_count", self._patch_labels(labels), accumulation)

                case _:
                    raise NotImplementedError(f"type {group.type} is not supported")

    def _patch_labels(self, base: list[(str, str)], extra: Iterable[(str, str)] = []) -> list[(str, str)]:
        pairs = ((key, str(value)) for key, value in self.labels.items() if value is not None)
        return sorted(dict(chain(base, pairs, extra)).items())


from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxDBExporter(MetricsExporter):
    def __init__(self, url: str, org: str, token: str, bucket: str) -> None:
        super().__init__()
        self.labels["iteration"] = 0
        self._org = org
        self._bucket = bucket
        self._client = InfluxDBClient(url=url, token=token, org=org)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def push_metrics(self, groups: Iterable[MetricsGroup]) -> None:
        timepoint = time.time_ns()
        self.labels["iteration"] += 1
        with self._client.write_api(write_options=SYNCHRONOUS) as api:
            def cast(args):
                name, suffix, labels, value = args
                point = Point(name + suffix).field("value", value).time(timepoint)
                return reduce(lambda x, label: x.tag(*label), labels, point)
            api.write(self._bucket, self._org, map(cast, self.parse_metrics_groups(groups)))


import torch
import torch_dipu


if __name__ == "__main__":
    org = "DIPU"
    bucket = "dipu-test"
    url = "THIS_IS_URL"
    token = "THIS_IS_TOKEN"
    exporter = InfluxDBExporter(url, org, token, bucket)

    for i in range(0, 1):
        limit = random.randrange(1000, 1000000)
        for nbytes in range(100, limit, 100):
            device = random.randrange(0, torch.cuda.device_count())
            with torch.cuda.device(device):
                raw = torch.empty(size=(nbytes,), dtype=torch.uint8, device='dipu')

        exporter.push_metrics(torch_dipu._C.metrics())
        print(f"iter: {i}")
        time.sleep(1)

    exporter.close()
    print('done')
```
