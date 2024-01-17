# DIPU python 测例

## 目录结构

- `unittests/`
  - `test_*.py`：所有简单测例。
  - `unittest_autogened_for_individual_scripts.py`：为独立测例自动生成的单元测试。
- `individual_scripts/`
  - `test_*.py`：所有独立测例脚本。
  - `generate_unittest_for_individual_scripts.py`：为独立测例自动生成单元测试代码的脚本。
- `utils/`：测试用的辅助小工具。
- `fix_needed/`：[#319](https://github.com/DeepLink-org/DIPU/pull/319) 中暂未整合的测例，后续将消除这个目录。
- `run_tests.sh`：运行所有自动化测例

## 如何添加测例

### 简单测例

简单测例是指使用默认 import 方式的测例，大部分 op 测例都应该是简单测例。简单测例应被添加到 `unittests/` 目录下，无需修改其他文件。

简单测例应包含一个继承了 `torch_dipu.testing._internal.common_utils.TestCase` 的测试类，该类继承了 `unittest.TestCase`，并添加了 `assertEqual`、`assertRtolEqual` 等测试函数的 tensor 支持。

#### 注意事项

- 测试**务必**做到自动化，所有的 `print` 应在提交前消除，也不要直接使用 `assert`。
  - gold 的构造有两种主要方式：手动构造，或由 torch 的 CPU 实现生成。
  - 对于难以进行 golden 测试的例子：
    - 对于带有随机性的 op，可以考虑考察其分布的特征（参考 multinomial、random 等）。
    - 可以考虑不使用 assertion，只检测 error 不检测 failure（加上注释说明）。
  - `torch.allclose` **不**检测 shape、dtype 等，请谨慎使用。
  - 如果需要检查 C++ 库内部的输出，可以使用 `utils.stdout_redirector.stdout_redirector` 来捕获。
  - 如果需要使用输出辅助 debug，可以考虑在使用 unittest 的 assertion 函数时传入 [`msg` 参数](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertEqual)。
- **请勿**做对全局空间有影响的事，例如：
  - 修改 import 库的内容；
  - 在全局空间中定义其他函数和变量（考虑挪至 class 内）；
  - 修改环境变量（可使用 `utils.local_eviron.local_eviron`）；
- 应根据 torch 的文档广泛地测试各种使用场景。
  - 尽量借助 setUp()、class 变量等方式简化代码，不要复制大量代码，以便后续维护。
- 对于预期会失败的测例，可以使用 `onlyOn` 和 `skipOn` 修饰器设置在某些设备上跳过测例（参考 cdist）。
  - 如果是 bug，请加上注释 `# TODO({user}): {plan ...}`。
- 代码风格请参照现有代码，并使用 `black` 进行格式化。

### 独立测例

独立测例是指依赖 import 方式的测例，大部分 rt 测例都应该是独立测例。独立测例应被添加到 `individual_scipts/` 目录下，无需修改其他文件。

独立测例应该是一个可独立运行的 python 脚本。这些测试脚本会被自动转为单元测试，脚本返回值为 0 说明测试成功，否则测试失败。

如果需要自动化检测 C++ 库内部的输出，可以使用 `utils.stdout_redirector.stdout_redirector` 来捕获。

独立测例可以包含 print。不过，在自动生成的单元测试中，独立测例中的输出会在测试通过的情况下被消除。

可以使用 `utils.test_in_subprocess.run_individual_test_cases` 在同一个文件中进行多个独立测例的编写。

#### 子进程的 coverage 收集

使用 `multiprocessing.Process` 创建的子进程在 CI 上跑 coverage 时不会被统计，因此使用这种测试方式（e.g. `test_allocator.py`）的独立测例需要一些特别的处理。

#### C++ `gcov`

~~在调用 `multiprocessing.Process` 之前，**必须**调用 `multiprocessing.set_start_method("spawn", force=True)` 修改 multiprocessing 的默认进程生成方式。~~

请使用 `utils.test_in_subprocess.run_individual_test_cases` 来创建子进程。

##### Python `coverage`

代码**无需**额外修改，而是需要在 coverage run 中加入额外的 flag。目前在 `run_tests.sh` 中已经配置好了。
