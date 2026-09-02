# lings — 纱线光谱异常检测

基于近红外采谱 CSV(1000~2250 nm,2501 点)的样本分析、可视化与深度学习分类项目。

> 🎓 第一次接触本项目的同学:请先看 **[docs/研究生入门指南.md](docs/研究生入门指南.md)**
> (环境安装、从零跑通、找代码资源、报错排查全覆盖)。

## 目录结构

```
lings/
├── data/                  # 原始数据
│   ├── 正常数据集/正常数据集/*.csv   (2461 个)
│   └── 异常数据集/异常数据集/*.csv   (254 个)
├── src/lings/             # 全部脚本
│   ├── templates/         # HTML 可视化模板
│   ├── analysis_spectral.py              # 数据加载/预处理/传统 ML 分析
│   ├── visualize_spectrum.py             # 单文件光谱可视化
│   ├── plot_all_spectra.py               # 全样本叠加图(正常蓝/异常黄)
│   ├── plot_wavelength_distribution.py   # 每波长分布带图/箱线图/统计 CSV
│   ├── plot_absorbance_histograms.py     # 指定波长吸光度直方图
│   ├── plot_absorbance_3d_field.py       # 3D 密度场 PNG
│   ├── build_absorbance_3d_html.py       # 交互式 3D 密度场 HTML(Plotly 内联)
│   ├── build_visualization.py            # 谱学可检测性 HTML
│   ├── build_absorbance_distribution.py  # 吸光度分布 HTML(2D)
│   ├── train_1d_cnn.py                   # 1D CNN 训练(随机 CV + 留一批次)
│   ├── train_1d_cnn_group_stress.py      # CNN 分组压力测试
│   └── train_multi_model.py              # 多模型对比训练(MLP/CNN/GRU/Transformer)
├── analysis/              # 中间分析结果 CSV/JSON
└── outputs/               # 全部产物:图、HTML、模型 pth、指标
    ├── multimodel/        # 多模型基础版结果
    └── multimodel_tuned/  # 不平衡调优版结果(推荐)
```

## 环境

uv 管理,虚拟环境在 `.venv`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt  # 首次
uv run python src/lings/<脚本>.py [参数]
```

## 常用命令

```bash
# 可视化
uv run python src/lings/plot_all_spectra.py                     # 全样本总览图
uv run python src/lings/build_absorbance_3d_html.py              # 可旋转 3D 密度场 HTML
open outputs/absorbance_3d_field.html

# 训练(多模型对比,含不平衡调优:过采样+噪声+F1 阈值)
uv run python src/lings/train_multi_model.py
```

## 数据说明

- 光谱 CSV 为分号分隔;头部为元数据,数据段以 `#;Wavelength / nm;Absorbance / AU` 开头
- 批次: DV 2199(主体)、DN 329、DW 128、DY 59(全正常);异常率约 9.4%,评估以
  AUPRC / G-mean / F1 / 召回率为准(全预测"正常"的准确率即 90.6%)
- 特征: SNV + Savitzky-Golay 一阶导,输入 `[N, 2, 601]`
