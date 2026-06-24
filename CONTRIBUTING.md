# 如何为 Open-AoE 贡献代码

> 本文档面向初次接触开源项目协作的贡献者，帮助你理解 Open-AoE 的项目结构、贡献流程和常见误区。

---

## 目录

- [0. 快速自查：你的贡献属于哪种类型？](#0-快速自查你的贡献属于哪种类型)
- [1. 项目结构约定](#1-项目结构约定)
- [2. 贡献流程](#2-贡献流程)
- [3. 代码规范](#3-代码规范)
- [4. License 与第三方依赖](#4-license-与第三方依赖)
- [5. 常见错误案例](#5-常见错误案例)
- [6. 检查清单](#6-检查清单)

---

## 0. 快速自查：你的贡献属于哪种类型？

在开始之前，先确认你的代码应该放在哪里：

| 你的贡献 | 应该放在 | 举例 |
|----------|----------|------|
| 某类任务的完整工具链（训练配方、可视化、仿真等） | 项目根目录下新建子目录 | `aoe-training-ready/`、`aoe-retarget-replay/` |
| 对某个已有子项目的功能增强 | 对应子目录内修改 | 在 `aoe-visualization/` 中增加新的渲染器 |
| 新的模型训练配方 | `aoe-training-ready/<model_name>/` | `aoe-training-ready/vitra/` |

**核心原则：不要修改不属于你的顶层文件。**（详见[第 5 节](#5-常见错误案例)）

---

## 1. 项目结构约定

### 1.1 顶层目录

```
Open-AoE/
├── README.md            # 项目总览（对外展示）
├── CONTRIBUTING.md      # 贡献指南（你正在读的文件）
├── CLAUDE.md            # AI 协作规范
├── LICENSE              # 项目许可证 (Apache 2.0)
├── LEGAL.md             # 第三方依赖许可声明
├── assets/              # 公共资源（下载脚本、共享配置）
├── open-aoe-2000h/      # 数据集文档与使用指南
├── aoe-visualization/   # 数据可视化工具
├── aoe-retarget-replay/ # Human-to-Robot 重映射工具
└── aoe-training-ready/  # 模型训练格式转换工具
```

### 1.2 子项目目录结构

每个子项目是一个**独立的功能模块**，必须包含：

```
<你的子项目名>/
├── README.md            # 子项目说明（必须）
├── LICENSE              # 许可证文件（可选，默认继承根 LICENSE）
├── pyproject.toml       # 或 requirements.txt —— 依赖声明（必须）
├── <源码目录>/           # 核心代码
├── scripts/             # CLI 入口脚本
├── configs/             # 配置文件
└── assets/              # 子项目自有的资源文件
```

**示例：已集成的子项目**

| 子项目 | 路径 | 功能 |
|--------|------|------|
| aoe-visualization | `aoe-visualization/` | AoE 数据可视化与渲染 |
| aoe-training-ready | `aoe-training-ready/` | 模型训练配方（GR00T、VITRA） |
| aoe-retarget-replay | `aoe-retarget-replay/` | 人机动作重映射 (Phantom) |
| open-aoe-2000h | `open-aoe-2000h/` | 数据集文档与下载 |

---

## 2. 贡献流程

### 2.1 准备工作

```bash
# 1. Fork 项目仓库（或获取分支权限）
# 2. 克隆仓库
git clone <仓库地址>
cd Open-AoE

# 3. 基于 main 分支创建你的功能分支
git checkout main
git pull origin main
git checkout -b <你的分支名>

# 分支命名建议：
#   feat/<功能名>        — 新功能
#   fix/<问题描述>       — Bug 修复
#   cooperate/<贡献者>   — 合作贡献分支
```

### 2.2 开发与提交

```bash
# 1. 在正确的目录下编写代码（见第 1 节）

# 2. 提交代码
git add <你的文件>
git commit -m "<类型>: <简短描述>"

# 提交信息格式建议：
#   feat: 添加 xxx 功能
#   fix: 修复 xxx 问题
#   doc: 更新 xxx 文档
#   chore: 清理 xxx 代码规范
```

### 2.3 提交 PR (Pull Request)

1. 将你的分支推送到远程仓库
2. 在 GitHub/GitLab 上创建 Pull Request，目标分支为 `main`
3. PR 描述中说明：
   - 这个贡献的功能是什么
   - 新增/修改了哪些文件
   - 依赖关系（是否需要额外的包或模型）
   - 如何验证功能正常

---

## 3. 代码规范

### 3.1 必须遵守

- **Copyright 头**：每个 Python 文件顶部必须有 SPDX 标识：

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

- **Docstring**：每个模块、类和公开函数需要有文档字符串
- **类型注解**：公开 API 的函数参数和返回值建议添加类型注解
- **README**：每个子项目必须有独立的 README.md

### 3.2 推荐遵守

- Python 代码使用 `ruff` 格式化
- 函数命名使用 snake_case，类名使用 PascalCase
- 不要在代码中硬编码绝对路径
- 使用 `pathlib` 而非 `os.path` 处理路径

### 3.3 禁止事项

- ❌ 不要修改 `inner/sample_data/` 中的数据文件
- ❌ 不要在 `` 中放置草稿或实验性代码
- ❌ 不要在一次提交中同时大幅修改多个顶层 `.md` 文件
- ❌ 不要擅自更改 28D joint space 定义：`[L_ARM(7), R_ARM(7), L_HAND(6), R_HAND(6), PAD(2)]`
- ❌ 不要直接提交二进制大文件（模型权重、视频、数据集等），使用外部下载脚本

---

## 4. License 与第三方依赖

### 4.1 项目许可证

Open-AoE 整体采用 **Apache 2.0** 许可证。你贡献的代码将默认以相同许可证发布。

### 4.2 第三方代码

如果你需要引用或集成第三方代码，请注意：

| 情况 | 处理方式 |
|------|----------|
| 代码可以 `pip install` 安装 | 写入 `requirements.txt` 或 `pyproject.toml`，**不要**复制源码到仓库中 |
| 代码只有少量修改才能用 | 在 `LEGAL.md` 中声明来源和许可证，保留原始版权声明 |
| 代码的许可证不兼容（如 CC-BY-NC-ND） | **不能**包含在开源发布中，只能让用户自行下载安装 |
| 大型模型权重文件（如 .pkl, .pt, .safetensors） | 提供下载脚本，**不要**直接提交到仓库 |

### 4.3 常见需要避免的第三方依赖

| 依赖 | 问题 | 正确做法 |
|------|------|----------|
| MANO 模型 (.pkl) | 需要用户注册，不可直接分发 | 使用 `assets/mano/download_mano.sh` 统一下载 |
| HaWoR | 含 CC-BY-NC-ND 代码，不可商用 | 列为 `pip install` 依赖，引用原始仓库 |
| 预训练模型权重 | 体积大，许可证各异 | 提供 HuggingFace 下载链接 |

### 4.4 新增第三方声明

如果你的贡献引入了新的第三方依赖，请在以下文件中添加相应条目：

1. **`LEGAL.md`** — 添加第三方组件声明，说明：
   - 依赖名称和用途
   - 许可证类型
   - 来源（URL 或 pip 包名）
2. **`README.md`** — 在「致谢」表格中添加项目引用和链接
3. **子项目 `README.md`** — 在文档中标注所引用的上游项目

---

## 5. 常见错误案例

> 以下案例基于实际协作中遇到的问题整理，目的是帮助后来者避免踩坑。

### 5.1 ❌ 错误：覆盖项目的顶层文件

**问题描述**：某贡献者将一个完整的独立项目直接复制到仓库中，覆盖了 `README.md`、`CLAUDE.md`，删除了 `STORY.md`、`PROJECT.md`、`report/` 和 `inner/` 目录。

**为什么不能这样做**：
- 这些顶层文件是**整个项目**的公共设施，不属于任何一个子项目
- 覆盖它们会破坏其他贡献者的工作（可视化、重映射、训练配方等）
- 后来合并时需要大量人工处理冲突，甚至无法合并

**正确做法**：
- 将你的代码放在项目根目录下对应的子目录中
- 只修改你负责的目录内的文件
- 如果确实需要更新顶层文件（如 `README.md`），只追加内容，不删除已有内容

### 5.2 ❌ 错误：直接提交第三方代码

**问题描述**：将 HaWoR、DROID-SLAM、Metric3D 等第三方项目的完整源码直接复制到仓库中。

**为什么不能这样做**：
- 这些代码有自己的许可证（如 CC-BY-NC-ND），可能与 Apache 2.0 不兼容
- 大量无关代码（CUDA kernel、C++ 源码、训练脚本）会让仓库膨胀
- 第三方项目独立更新后，你的副本会过时，且无法自动同步

**正确做法**：
- 将第三方依赖写入 `requirements.txt` 或 `pyproject.toml`
- 在 README 中说明安装步骤
- 如果确实需要少量修改，只保留修改部分（patch），并在 `LEGAL.md` 中声明

### 5.3 ❌ 错误：提交模型权重文件

**问题描述**：将 MANO 模型 `.pkl` 文件、语言嵌入 `.pt` 文件等直接提交到仓库。

**为什么不能这样做**：
- MANO 模型需要用户在官网注册才能合法使用，直接分发违反其许可证
- 模型权重文件通常很大（几 MB 到几 GB），会让仓库体积迅速膨胀
- 用户克隆仓库时被迫下载不需要的文件

**正确做法**：
- 提供下载脚本（参考 `assets/mano/download_mano.sh`）
- 在 README 中说明如何获取
- 将文件路径加入 `.gitignore`

### 5.4 ❌ 错误：代码缺少 Copyright 头和 License 声明

**问题描述**：整个子项目没有 LICENSE 文件，所有 Python 文件没有 SPDX 版权声明。

**为什么不能这样做**：
- 开源发布需要明确的许可证，否则用户无法合法使用
- 缺少版权声明让人无法追溯代码来源和责任人

**正确做法**：
- 每个 `.py` 文件顶部添加 SPDX 头（见第 3.1 节）
- 子项目目录下添加 LICENSE 文件（或继承根目录的 LICENSE）

### 5.5 ❌ 错误：在分支中提交个人数据

**问题描述**：提交了个人 symlink（如 `datasets -> /home/xxx/datasets`）、本地输出目录、IDE 配置文件等。

**正确做法**：
- 检查 `.gitignore` 是否已覆盖你的本地文件
- 使用 `git status` 确认只提交了需要的文件
- 不要提交任何包含绝对路径的文件

---

## 6. 检查清单

提交 PR 前，请逐项确认：

### 代码质量

- [ ] 所有 `.py` 文件有 SPDX 版权声明头
- [ ] 模块和公开函数有 docstring
- [ ] 公开 API 有类型注解
- [ ] 子项目有独立的 README.md

### 项目结构

- [ ] 代码放在正确的目录下（项目根目录对应子目录）
- [ ] 没有修改或删除顶层文件
- [ ] 没有修改其他子项目的文件

### 依赖与 License

- [ ] 第三方依赖已写入 `requirements.txt` 或 `pyproject.toml`
- [ ] 没有直接提交第三方源码（除非有兼容的许可证且写入 `LEGAL.md`）
- [ ] 没有提交模型权重文件（使用下载脚本代替）
- [ ] 新增的第三方依赖已在 `LEGAL.md` 中声明，并在 `README.md` 致谢表格中添加引用

### 文件管理

- [ ] 没有提交二进制大文件（视频、模型、数据集）
- [ ] 没有提交个人数据或本地路径
- [ ] `.gitignore` 已覆盖应忽略的文件
- [ ] 提交信息清晰描述了变更内容

### 验证

- [ ] 代码可以在干净的 Python 环境中运行
- [ ] README 中的安装和使用说明准确无误
- [ ] 如果是新子项目，已更新 `aoe-training-ready/README.md` 中的表格

---

## 附录：快速参考

### 分支命名

```
feat/<功能名>          # 新功能分支
fix/<问题描述>         # Bug 修复
cooperate/<名称>       # 合作贡献分支
```

### 提交信息格式

```
<类型>: <简短描述>

feat: 添加 G1 机器人 IK 解算器
fix: 修复 MANO FK 左手坐标系转换错误
doc: 更新 README 中的安装说明
chore: 添加 SPDX 版权声明头
```

### 常用命令

```bash
# 检查将要提交的文件
git status

# 查看与 main 的差异
git diff main --stat

# 查看完整的 diff
git diff main

# 提交前检查是否有大文件
find . -type f -size +5M -not -path "./.git/*"
```

---

> 如有疑问，请在项目中提 Issue 或联系项目维护者。
> 感谢你为 Open-AoE 做出的贡献！