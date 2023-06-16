# 开发者指南

## 贡献代码

欢迎加入 DIPU 社区，我们致力于连接 PyTorch 深度学习框架和标准算子接口，使得 PyTorch 框架能够在人工智能芯片上做训练、推理。我们欢迎任何类型的贡献，包括但不限于

**修复错误**

修复代码实现错误的步骤如下：

1. 如果提交的代码改动较大，建议先提交 issue，并正确描述 issue 的现象、原因和复现方式，讨论后确认修复方案。
2. 修复错误并补充相应的单元测试，提交拉取请求。

**新增功能或组件**

1. 如果新功能或模块涉及较大的代码改动，建议先提交 issue，确认功能的必要性。
2. 实现新增功能，提交拉取请求。

**文档补充**

修复文档可以直接提交拉取请求

添加文档或将文档翻译成其他语言步骤如下

1. 提交 issue，确认添加文档的必要性。
2. 添加文档，提交拉取请求。

## 贡献流程

#### 拉取请求工作流

如果你对拉取请求不了解，没关系，接下来的内容将会从零开始，一步一步地指引你如何创建一个拉取请求。如果你想深入了解拉取请求的开发模式，可以参考 github [官方文档](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

#### 1. 复刻仓库

当你第一次提交拉取请求时，先复刻仓库原代码库，这里以 dipu 为例。点击 GitHub 页面右上角的 **Fork** 按钮，复刻后的代码库将会出现在你的 GitHub 个人主页下。
将代码克隆到本地

```shell
git clone git@github.com:{username}/dipu.git
```

添加原代码库为上游代码库

```bash
git remote add upstream git@github.com::DeepLink-org/dipu
```

检查 remote 是否添加成功，在终端输入 `git remote -v`

```bash
origin	git@github.com:{username}/dipu.git (fetch)
origin	git@github.com:{username}/dipu.git (push)
upstream	git@github.com:DeepLink-org/dipu (fetch)
upstream	git@github.com:DeepLink-org/dipu (push)
```

> 这里对 origin 和 upstream 进行一个简单的介绍，当我们使用 git clone 来克隆代码时，会默认创建一个 origin 的 remote，它指向我们克隆的代码库地址，而 upstream 则是我们自己添加的，用来指向原始代码库地址。当然如果你不喜欢他叫 upstream，也可以自己修改，比如叫 dipu。我们通常向 origin 提交代码（即 fork 下来的远程仓库），然后向 upstream 提交一个 pull request。如果提交的代码和最新的代码发生冲突，再从 upstream 拉取最新的代码，和本地分支解决冲突，再提交到 origin。


#### 2. 创建开发分支

我们需要基于 main 创建开发分支，建议的分支命名规则为 `username/pr_name`。

```shell
git checkout -b yhc/refactor_contributing_doc
```

在后续的开发中，如果本地仓库的 main 分支落后于 upstream 的 main 分支，我们需要先拉取 upstream 的代码进行同步，再执行上面的命令

```shell
git pull upstream main
```


#### 3. 推送代码到远程
将代码推送到远程仓库，如果是第一次推送，可以在 `git push` 后加上 `-u` 参数以关联远程分支

```shell
git push -u origin {branch_name}
```

这样下次就可以直接使用 `git push` 命令推送代码了，而无需指定分支和远程仓库。


#### 4. 提交拉取请求（PR）

(1) 在 GitHub 的 Pull request 界面创建拉取请求


(2) 根据指引修改 PR 描述，以便于其他开发者更好地理解你的修改

描述规范详见[拉取请求规范](#拉取请求规范)

&#160;

**注意事项**

(1) PR 描述应该包含修改理由、修改内容以及修改后带来的影响，并关联相关 Issue（具体方式见[文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)）

(2) 如果是第一次为 dipu 做贡献，需要签署 CLA

(3) 检查提交的 PR 是否通过 CI（集成测试）


(4) 如果 PR 通过了 CI，那么就可以等待其他开发者的 review，并根据 reviewer 的意见，修改代码，并重复 [3](#3-提交代码进行ci验证)-[4](#4-代码review分支合并) 步骤，直到 reviewer 同意合入 PR。

所有 reviewer 同意合入 PR 后，我们会尽快将 PR 合并到主分支。

#### 5. 解决冲突

随着时间的推移，我们的代码库会不断更新，这时候，如果你的 PR 与主分支存在冲突，你需要解决冲突，解决冲突的方式有两种：

```shell
git fetch --all --prune
git rebase upstream/main
```

或者

```shell
git fetch --all --prune
git merge upstream/main
```

如果你非常善于处理冲突，那么可以使用 rebase 的方式来解决冲突，因为这能够保证你的 commit log 的整洁。如果你不太熟悉 `rebase` 的使用，那么可以使用 `merge` 的方式来解决冲突。

### 拉取请求规范

1. 一个`拉取请求`对应一个短期分支

2. 粒度要细，一个`拉取请求`只做一件事情，避免超大的`拉取请求`

   - Bad：一个PR里补充多个模型所需的所有算子
   - Acceptable：一个PR里实现一个或几个相关算子
   - Good：修复某个算子 input 为空时引发的 bug

3. 每次 commit 时需要提供清晰且有意义 commit 信息

4. 提供清晰且有意义的`拉取请求`描述

   - 标题写明白任务名称，一般格式:\[Prefix\] Short description of the pull request (Suffix)
   - prefix: 新增功能 \[Feature\], 修 bug \[Fix\], 文档相关 \[Docs\], 开发中 \[WIP\] (暂时不会被review)
   - 描述里介绍`拉取请求`的主要修改内容，结果，以及对其他部分的影响, 参考`拉取请求`模板
   - 关联相关的`议题` (issue) 和其他`拉取请求`

5. 如果引入了其他三方库，或借鉴了三方库的代码，请确认他们的许可证和 dipu License 兼容，并在借鉴的代码上补充 `This code is inspired from http://`