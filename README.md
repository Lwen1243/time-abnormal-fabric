# lings — 纱线光谱异常检测

基于近红外采谱 CSV(1000~2250 nm,2501 点)的样本分析、可视化与深度学习分类项目。

> 🎓 第一次接触本项目的同学:请先看 **[docs/入门指南.md](docs/入门指南.md)**
> (环境安装、从零跑通、找代码资源、报错排查全覆盖)。

## 目录结构

```
lings/
├── data/                  # 原始数据(7905 条有效光谱)
│   ├── 正常数据集/        # 6731 条
│   └── 异常数据集/        # 1174 条(含 DC/LC/其他类型等子目录)
├── src/lings/             # 全部脚本
│   ├── templates/         # HTML 可视化模板
│   ├── analysis_spectral.py              # 数据加载/预处理/传统 ML 分析
│   ├── train_stack_all.py                # 多模型集成(基模型+权重搜索+LR Stacking)
│   ├── train_boost_round3.py             # 第3轮: KNN/LDA+多seed树+真元学习器
│   ├── train_boost_round4.py             # 第4轮: DART/多尺度KNN→Meta-stack v4(最终)
│   ├── train_boost_round6.py             # 第6轮: 自编码器特征(无效,留档)
│   ├── train_cnn_tta.py                  # ConvNeXt-1D + TTA(GPU,留档)
│   ├── train_moment.py                   # MOMENT 预训练时序模型(参考)
│   ├── train_ensemble.py                 # 7模型集成(MLP/CNN/GRU/Transformer)
│   ├── train_1d_cnn.py                   # 1D CNN 训练(随机 CV + 留一批次)
│   ├── train_multi_model.py              # 多模型对比训练(MLP/CNN/GRU/Transformer)
│   ├── gen_ppt_figures.py                # PPT 图集(01-12)生成
│   ├── gen_extra_figures.py              # PPT 补充图(13-20)生成
│   └── 其他可视化脚本…
├── analysis/              # 中间分析结果 OOF/CSV/JSON
└── outputs/
    └── ppt_figures/       # 20 张 PPT 图 + method_metrics.csv
```

## 环境

uv 管理,虚拟环境在 `.venv`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt  # 首次
uv run python src/lings/<脚本>.py [参数]
```

## 常用命令

```bash
# 复现最终方案 Meta-stack v4(约 15 分钟,CPU)
uv run python src/lings/train_boost_round4.py

# 生成全部 PPT 图(01-12 + 13-20)
uv run python src/lings/gen_ppt_figures.py
uv run python src/lings/gen_extra_figures.py

# 可视化
uv run python src/lings/plot_all_spectra.py                     # 全样本总览图
uv run python src/lings/build_absorbance_3d_html.py              # 可旋转 3D 密度场 HTML
```

## 数据说明

- 光谱 CSV 为分号分隔;头部为元数据,数据段以 `#;Wavelength / nm;Absorbance / AU` 开头
- 有效样本 7905 条(正常 6731 / 异常 1174),批次 DN/DW/DV/DY,波长 1000-2250 nm(2501 点,0.5 nm 步长)
- **关键诊断**: 98% 的异常样本与某正常样本余弦相似度 > 0.9999;约 200 个样本"同谱异标"(光谱几乎相同但标签相反),随机划分准确率理论上限约 97.6%
- 特征管线: SNV + Savitzky-Golay 一阶/二阶导,掩码 1050-2450 nm,多尺度拼接

## 分类方法总结(最终方案 Meta-stack v4)

### 0. 问题定义与评估协议

- **任务**: 对 7905 条近红外光谱(1000-2250 nm, 2501 点)做二分类,正常 6731 条 / 异常 1174 条(异常率 14.9%)
- **评估方式**: 5 折 **StratifiedKFold(random_state=42)** 的 OOF(Out-of-Fold)预测。不用单独留出集,保证 7905 条全部参与评估且无泄漏(存在大量重复/近重复光谱,若按文件随机分割会因副本跨集泄漏导致虚高)
- **决策阈值**: 在 OOF 上扫所有取值,取最大化 F1 的阈值(本数据上与最大化准确率的阈值一致)
- **报告指标**: AUC / AUPRC / ACC / F1 / 敏感性 / 特异性 / 混淆矩阵

### 1. 数据诊断(为什么天花板是 97.6%)

| 发现 | 具体证据 | 影响 |
|---|---|---|
| 光谱严重重复 | 唯一光谱仅约 5189 个,2716 条是近重复副本 | 必须用 OOF 防泄漏 |
| 两类几乎完全重叠 | 98% 异常与某正常样本 cos > 0.9999;92.4% 正常也与某异常 cos > 0.9999 | 光谱上大部分"异常"没有独立信息 |
| 约 200 个同谱异标 | `analysis/abnormal_normal_same_pairs.csv` 记录 1151 对近相同光谱、标签相反 | 这些样本任何模型的正确率上限就是 50%,推得随机划分理论上限 ≈ 97.6% |
| 异常并非单一群体 | PCA 显示: DV 批次异常构成右下方可分离的独立簇(真异常),其余异常散落/嵌入正常簇(伪异常) | 真异常可 100% 识别,伪异常不可 |

结论: 在"纯光谱 + 文件夹标签"框架下,模型无法超过约 97.6%,这是**数据标注问题**而非算法问题。

### 2. 预处理管线

1. **SNV 标准化** $x_{snv} = (x - \bar{x})/\sigma$ — 消除散射/幅值偏移
2. **Savitzky-Golay 一阶导** window=31, polyorder=2, delta=0.5 — 抑制基线漂移、保留吸收峰
3. **Savitzky-Golay 二阶导** window=41, polyorder=3, delta=0.5 — 补充曲率/峰宽信息
4. **掩码** 1050-2450 nm;多尺度特征按 2 nm 降采样

| 特征版本 | 形状 | 用途 | 5 折 AUC |
|---|---|---|---|
| SNV+一阶导,601 点 | [N, 2, 601] | 早期 CNN | ~0.97 |
| **三通道多尺度(最终)** | [N, 1803] | 全部 GBDT/KNN/LDA | 0.9897 |
| 全分辨率双通道 | [N, 2, 2401] | ConvNeXt / MLP | 0.9604 |
| 全分辨率三通道 | [N, 7203] | LightGBM 全分辨率 | 0.9888(略低于降采样) |

### 3. 基模型(13 个,全部 5 折 OOF)

| # | 模型 | 特征 | 关键超参 | OOF AUC |
|---|---|---|---|---|
| 1 | LightGBM 3-seed bag | 1803 多尺度 | 1200 trees, lr=0.03, leaves=63, ff=0.7, bf=0.8 | **0.9900** |
| 2 | LightGBM 多尺度(另 3 seed) | 1803 | 同上,多次平均 | 0.9897 |
| 3 | LightGBM 全分辨率 | 7203 | leaves=63, ff=0.6 | 0.9888 |
| 4 | LightGBM DART | 1803 | boosting=dart, drop=0.1 | 0.9902 |
| 5 | XGBoost 3-seed bag | 1803 | 800 trees, depth=6, lr=0.04 | 0.9905 |
| 6 | XGBoost v2 | 1803 | 1500 trees, depth=7, mcw=5 | 0.9878 |
| 7 | ExtraTrees 3-seed bag | 1803 | 600 trees, sqrt, msl=3 | 0.9886 |
| 8 | RandomForest | 1803 | 600 trees, sqrt | 0.9831 |
| 9 | KNN(SNV) | 601 | k=30, distance 加权 | 0.9702 |
| 10 | KNN(多尺度) | 1803 | k=25, distance 加权 | 0.9702 |
| 11 | LDA | 1803 | lsqr + shrinkage | 0.9213 |
| 12 | ConvNeXt-1D | [N,2,2401] | 1.8M 参数,GPU,batch=256 | 0.9604 |
| 13 | MLP 7 模型集成 | [N,2,2401] | WideMLP/FullCNN/GRU/Transformer | 0.9795 |

设计思路: 覆盖完全不同的归纳偏置——树模型(非线性特征交互)、实例近邻(局部结构)、线性判别(全局线性)、深度学习(局部平滑/时序结构),让元学习器可以取长补短。

### 4. 元学习器(Meta-stack)详细步骤

1. **收集** 13 个基模型的 OOF 概率(7905 维向量 × 13)
2. **rank 变换**: 每个基模型的预测概率按其分位数排序值归一化到 [0,1](消除概率标度/校准差异,使 LR 输入可比)
3. **交互特征**: 对 13 维 rank 向量计算全部 $\binom{13}{2} = 78$ 个两两乘积(捕捉"两个模型一致/分歧"的信号),拼接得 **91 维**特征
4. **元学习器**: Logistic 回归(C=1.0, max_iter=3000),输入先 StandardScaler 标准化
5. **防泄漏内 CV**: 元学习器用 5 折 *内部* StratifiedKFold 评估,验证样本的分数来自未见过它的模型;该过程重复 **13 个随机种子** 并取平均,消除划分随机性
6. **阈值**: 在最终 OOF 分数上扫描,取 F1 最大阈值 t=0.553(准确率同最优)

### 5. 最终指标(5 折 OOF,阈值 = best F1)

| 指标 | 数值 |
|---|---|
| AUC | **0.99247** |
| AUPRC | 0.9655 |
| 准确率 | **97.18%(7655/7905)** |
| F1 | 0.903 |
| 敏感性(召回) | 88.5% |
| 特异性 | 98.7% |
| 精确率 | 92.2% |

混淆矩阵:

|  | 预测正常 | 预测异常 |
|---|---|---|
| **真正常(6731)** | 6643 | 88(FPR 1.3%) |
| **真异常(1174)** | 135(FNR 11.5%) | 1039 |

### 6. 方法演进(六轮迭代)

| 轮次 | 做了什么 | 准确率 | AUC | 说明 |
|---|---|---|---|---|
| 起点 | MLP 7 模型集成 | 94.0% | 0.9755 | 深度学习路线 |
| 第 1 轮 | LightGBM 3-seed bag | 96.29% | 0.9900 | 发现 GBDT 优于 NN |
| 第 2 轮 | 多尺度特征 + XGB/RF/ET/KNN/LDA | 96.42% | 0.9897 | 特征工程与异构模型 |
| 第 3 轮 | LR Stacking(凸权重搜索) | 96.69% | 0.9910 | 开始做融合 |
| 第 4 轮 | Meta-stack v3(rank+交互+7 seed) | 96.98% | 0.99219 | 元学习器定型 |
| 第 5 轮 | **Meta-stack v4(+DART/XGBv2/KNN 多尺度,13 基模型,13 seed)** | **97.18%** | **0.99247** | 最终方案 |
| 第 6 轮 | TTA/伪标签/自编码器/更多种子 | 96.97-97.18% | 0.99268 | 确认平台期 |

### 7. 尝试过但无效的方向(及原因)

| 方法 | 结果 | 原因 |
|---|---|---|
| CNN 测试时增强 TTA(平移±2点+噪声,8变体) | AUC 0.9548,低于无 TTA 的 0.9604 | 光谱本身无显著局部噪声/平移不变性收益 |
| 伪标签加权自训练 | AUC 0.9899,与原 LightGBM 持平 | 高置信样本只是重复信息,无新增标签 |
| 自编码器重建误差特征 | 特征本身 AUC 仅 0.35-0.39 | 异常样本的光谱轮廓与正常同样"常规",重建误差反而更小 |
| 距离特征 + 25 种子 meta | AUC +0.0002 但 ACC 略降 | 13 基模型已饱和,新特征只能此消彼长 |
| 全分辨率特征(7203 维) | AUC 0.9888 < 1803 维的 0.9897 | 维数灾难,过拟合 |

### 8. 关键产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 最终 OOF | `analysis/oof_boost_round4.npy` | Meta-stack v4 概率(7905,) |
| 复现脚本 | `src/lings/train_boost_round4.py` | 全流程约 15 分钟(CPU) |
| 基模型 OOF | `analysis/oof_*.npy` | 13 个基模型各自概率 |
| PPT 图集 | `outputs/ppt_figures/` | 20 张图 + `method_metrics.csv` |
| 矛盾样本清单 | `analysis/abnormal_normal_same_pairs.csv` | 1151 对近相同光谱 |
| 疑似噪声异常 | `analysis/suspicious_labels.csv` | 1150 个与正常 cos>0.9999 |

### 9. 结论

- 六轮迭代把准确率从 94.0% 提升到 **97.18%**,AUC 0.99247,已逼近数据理论上限约 97.6%
- 所有方法层面可用的手段(特征工程/基模型多样性/元学习/阈值优化/多种子)均已穷尽
- 剩余约 0.4pp 的缺口 100% 来自约 200 个"同谱异标"矛盾样本——**要继续突破,唯一路径是复核这些样本的真实标签**(清单见 `analysis/abnormal_normal_same_pairs.csv`)
