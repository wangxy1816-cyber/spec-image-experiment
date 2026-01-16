# LIBS 图谱融合实验 (Spec-Image Experiment)

本项目用于博士生研究项目中的图谱融合算法测试。

## 项目结构
- `src/main.py`: 主程序入口，在此配置数据地址。
- `src/spec_image_experiment/my_module.py`: 核心算法库，包含三种实验逻辑。

## 实验运行
1. 安装依赖：`uv sync`
2. 修改 `src/main.py` 中的 `PATH_ROOT` 指向你的本地数据集路径。
3. 运行：`uv run src/main.py`