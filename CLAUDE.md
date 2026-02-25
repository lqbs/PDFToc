# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PDFdir 是一个 PDF 导航书签添加工具，根据已有的目录文本为 PDF 自动生成导航书签（大纲）。支持 GUI 和 CLI 两种操作方式。

**项目仓库**: https://github.com/chroming/pdfdir

## 项目结构

```
PDFToc/
├── run_gui.py           # GUI 启动脚本
├── run_cli.py           # 命令行接口
├── run.py               # 基础运行脚本
├── config.ini           # 配置文件（层级正则表达式）
├── requirements.txt     # 核心依赖
├── requirements_dev.txt # 开发依赖
├── src/
│   ├── pdfdirectory.py  # 主 API 接口
│   ├── convert.py       # 目录文本转换为索引字典
│   ├── config.py        # 配置管理
│   ├── version.py       # 版本管理
│   ├── updater.py       # 更新检查
│   ├── gui/             # PyQt5 GUI 模块
│   ├── pdf/             # PDF 操作模块
│   │   ├── pdf.py       # PDF 读写
│   │   └── bookmark.py  # 书签添加
│   └── language/        # 多语言文件
└── tests/
    └── test_convert.py  # 转换模块测试
```

## 核心模块说明

### 1. `src/convert.py` - 目录文本解析

核心功能是将目录文本转换为结构化的索引字典。

**主要函数**:
- `split_page_num(text)`: 从文本中分离标题和页码，支持多种括号格式
- `check_level(title, level0, level1, ...)`: 检查目录项的层级
- `convert_dir_text(...)`: 主转换函数，返回索引字典
- `generate_level_pattern_by_prefix_space(dir_list)`: 通过前缀空格自动生成层级模式

### 2. `src/pdfdirectory.py` - 主 API

提供 `add_directory()` 函数，作为整个工具的核心 API：
- 调用 `convert_dir_text()` 解析目录文本
- 调用 `add_bookmark()` 将书签写入 PDF

### 3. `src/pdf/bookmark.py` - PDF 书签添加

处理 PDF 文件的书签写入，支持保留原 PDF 的注释和大纲。

### 4. GUI 模块 (`src/gui/`)

- `main.py`: 主窗口控制器
- `main_ui.py`: UI 界面代码（从 main_ui.ui 生成）
- 支持拖拽调整目录顺序和层级
- 多语言支持（中文/英文）

## 环境配置

### 依赖安装

```bash
# 核心依赖
pip install -r requirements.txt

# 开发依赖（打包等）
pip install -r requirements_dev.txt
```

**依赖列表**:
- `requests~=2.32.3`
- `pypdf[crypto]~=3.17.0`
- `six~=1.16.0`
- `PyQt5~=5.15.7`

## 常用命令

### 运行程序

```bash
# GUI 模式（推荐）
python run_gui.py

# 命令行模式
python run_cli.py <pdfPath> <tocPath> [options]

# 基础模式
python run.py
```

### 命令行使用示例

```bash
# 基本用法
python run_cli.py input.pdf toc.txt --offset 0

# 自定义层级正则
python run_cli.py input.pdf toc.txt --offset 0 \
  --l0 "^\d+\.\s?" \
  --l1 "^\d+\.\d+\w?\s?" \
  --l2 "^\d+\.\d+\.\d+\w?\s?"
```

**CLI 参数**:
- `pdfPath`: PDF 文件路径
- `tocPath`: 目录文本文件路径
- `--offset`: 页码偏移量
- `--l0` ~ `--l5`: 各级目录的正则表达式

### 运行测试

```bash
# 运行所有测试
pytest tests/test_convert.py -v

# 运行特定测试
pytest tests/test_convert.py::test_convert_dir_text -v
```

### 打包程序

使用 PyInstaller 打包：

```bash
pyinstaller -D run_gui.py -i "pdf.ico" --exclude config.ini --distpath . -n "pdfdir" --noconsole
```

## 配置文件

### config.ini

存储层级正则表达式配置：

```ini
[LEVEL]
l1 = "^\d+\.\s?"
l2 = "^\d+\.\d+\w?\s?"
l3 = "^\d+\.\d+\.\d+\w?\s?"
...
```

## 开发要点

### 目录文本格式

程序处理的目录文本格式为：**标题+页码+换行符**

```
中译版序言
致中国读者
前言
第1章 社会心理学导论 2
第一编 社会思维
第2章 社会中的自我 32
...
```

页码通过正则匹配文本结尾处的数字，支持多种括号格式：`()`, `[]`, `{}`, `<>`, `（）`, `【】`, `「」`, `《》`。

### 层级识别

支持 6 级目录层级，可通过以下方式识别：
1. 自定义正则表达式（配置在 config.ini 或通过 CLI 参数）
2. 前缀空格自动识别（`level_by_space=True`）

### GUI 开发

- UI 设计文件：`src/gui/main_ui.ui`（使用 Qt Designer 编辑）
- 转换 UI 为 Python 代码：`src/gui/ui_to_py.py` 或 `ui_to_py.bat`

## CI/CD

项目使用 GitHub Actions 自动化构建：
- Windows: `.github/workflows/windows-release.yml`
- macOS: `mac-release.yml`, `mac-silicon-release.yml`, `mac-py310-release.yml`
- Linux: `linux-release.yml`

## 已知问题

1. 非正文部分（序言、目录等）无页码时默认链接到第一页
2. macOS 创建 config.ini 时可能有权限问题（已在代码中注释禁用）
3. 无页码的目录项会链接到上一个有页码的标题页
