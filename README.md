# 化学实验动作识别与监控系统 (Chemical Experiment Action Recognition & Monitoring System)

本项目是一个基于计算机视觉和多模态大模型的化学实验动作识别与监控系统。通过 YOLO 目标检测和 Qwen-VL 大模型分析，实时监控实验过程，自动识别实验动作，并验证操作是否符合标准作业程序 (SOP)。

## 主要功能

*   **实时动作识别**: 利用 YOLO 模型识别实验器材（如烧杯、试管、滴管等）和关键动作（如拿取、倾倒、搅拌、滴加等）。
*   **流程状态监控**: 使用有限状态机 (FSM) 跟踪实验步骤，实时显示当前实验进度。
*   **AI 智能分析**: 集成 Qwen-VL 多模态大模型，对实验视频片段进行深度分析，验证操作规范性。
*   **可视化界面**: 基于 Streamlit 构建的 Web 界面，提供实时的视频流显示、步骤状态卡片和分析报告。
*   **实验报告生成**: 自动记录实验过程，生成包含动作序列和合规性检查的实验报告。

## 环境要求

请确保已安装 Python 3.8+ 环境。

### 依赖库

主要依赖库包括：
*   streamlit
*   opencv-python
*   numpy
*   ultralytics (YOLO)
*   Pillow
*   openai (用于调用 Qwen-VL API)

## 安装指南

1.  克隆本项目到本地：
    ```bash
    git clone <repository_url>
    cd Chemical_0204
    ```

2.  安装依赖：
    建议使用 `pip` 安装所需的 Python 包。
    ```bash
    pip install streamlit opencv-python numpy ultralytics Pillow openai
    ```

3.  模型准备：
    *   确保 `weights/` 目录下包含训练好的 YOLO 模型文件（例如 `yolo11m-best01222.pt`）。
    *   确保 Qwen-VL API 服务已启动或可访问（配置见 `qwen_api.py`）。

## 使用说明

1.  **启动应用**：
    在项目根目录下运行以下命令启动 Streamlit 应用：
    ```bash
    streamlit run app.py
    ```

2.  **操作界面**：
    *   打开浏览器访问显示的本地地址 (通常是 `http://localhost:8501`)。
    *   系统将自动加载实验配置，并在界面左侧显示实验步骤列表。
    *   主界面将显示视频流或上传视频进行分析。

3.  **实验监控**：
    *   系统会自动检测视频中的动作并更新步骤状态（待开始 -> 进行中 -> 完成/失败）。
    *   每个步骤完成后，会调用 Qwen-VL 进行复核，并在界面上显示分析结果。

## 项目结构

```text
Chemical_0204/
├── app.py                  # Streamlit 应用主入口
├── monitor-method.py       # 核心监控逻辑，包含动作识别和空间关系计算
├── LabFSM.py               # 实验流程有限状态机实现
├── qwen_api.py             # Qwen-VL 大模型 API 接口封装
├── generate_step.py        # 步骤生成与管理工具
├── weights/                # 存放 YOLO 模型权重文件
├── .streamlit/             # Streamlit 配置文件
├── process_structure.json  # 实验流程结构定义
└── steps.json              # 详细步骤定义
```

## 配置说明

*   **实验步骤**: 修改 `process_structure.json` 或 `steps.json` 可调整实验流程和SOP描述。
*   **模型配置**: 在 `app.py` 和 `monitor-method.py` 中可以调整 YOLO 模型路径和检测阈值。
*   **API 配置**: 在 `qwen_api.py` 中配置 Qwen-VL 的 `OPENAI_API_BASE` 和 `OPENAI_API_KEY`。

## 注意事项

*   请确保运行环境具备 GPU 支持，以获得流畅的 YOLO 检测和视频处理体验。
*   Qwen-VL 服务需要单独部署或确保存储 API 地址可访问。
