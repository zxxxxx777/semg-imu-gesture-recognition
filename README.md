# sEMG–IMU Gesture Recognition

本仓库收录基于表面肌电（sEMG）和运动传感器信号的手势识别实验，覆盖 NinaPro DB2、DB5、DB7 与 EMG-EPN-107。每组实验包含单模态基线、特征融合基线，以及 `TEST_*` 目录中的双分支交叉注意力融合模型；训练流程包括跨受试者预训练、目标受试者微调和测试。仓库中的 CSV 是已有实验的结果摘要，不包含原始数据或模型权重。

## 项目结构

```text
DB2/  DB5/  DB7/  EPN/
  baseline_*/       sEMG + 运动传感器特征融合基线
  baseline_*_SEMG/  仅使用 sEMG 的基线
  baseline_*_IMU/   仅使用运动传感器的基线
  TEST_*/           双分支交叉注意力融合实验
```

各实验目录主要包含 `config.py`（路径、受试者和训练参数）、`utils.py`（数据读取与窗口处理）、`data.py`（数据集）、`models.py`（网络）、`train.py`（训练与评估），以及入口 `main_fusion_finetune.py`。对应的 `*_summary.csv` 或 `summary.csv` 保存实验汇总。不同数据集的运动传感器字段不同：DB2/DB5 使用加速度信号，DB7 使用加速度、陀螺仪和磁力计，EPN 使用陀螺仪、加速度与四元数。

## 获取数据集

| 数据集 | 官方获取地址 | 本项目使用的数据 |
| --- | --- | --- |
| NinaPro DB2 | [DB2 官方下载页](https://ninapro.hevs.ch/instructions/DB2.html) | `.mat` 预处理文件；默认使用 E1、E2 动作集 |
| NinaPro DB5 | [DB5 官方下载页](https://ninapro.hevs.ch/instructions/DB5.html) | `.mat` 预处理文件；默认使用 E2、E3 动作集 |
| NinaPro DB7 | [DB7 官方下载页](https://ninapro.hevs.ch/instructions/DB7.html) | `.mat` 预处理文件；代码读取 E1、E2 文件 |
| EMG-EPN-107 | [Zenodo 数据集页面](https://zenodo.org/records/19829636) | `Dataset_EMG-EPN-107.rar`；代码筛选 200 Hz sEMG 用户 |

下载并解压后，按实际路径修改**所运行实验目录**中的 `config.py`。NinaPro 代码默认寻找如下文件（以 DB2 为例）：

```text
<LOSO_BASE_DATA_DIR>/Subject_1/S1_E1_A1.mat
<LOSO_BASE_DATA_DIR>/Subject_1/S1_E2_A1.mat
...
```

DB5 的文件由 `EXERCISES` 决定，DB7 默认读取 `S{subject}_E1_A1.mat` 和 `S{subject}_E2_A1.mat`。下载包的目录名称若不同，请整理为代码预期的结构，或调整对应的路径和文件名。EPN 代码默认寻找 `<DATA_ROOT>/training/user_001/userData.mat` 等文件，并从 `userData` 读取信号；需保留下载包中相应的层级，必要时调整 `DATA_ROOT` 和 `OUTER_TRAIN_GROUP`。请遵守原始数据集页面的使用与引用要求。

## 安装与运行

建议使用 Python 虚拟环境。核心依赖来自仓库中 Python 文件的第三方 `import`；PyTorch 如需 GPU 支持，请按本机 CUDA 环境安装合适的版本。

```bash
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -r requirements.txt
```

运行前检查目标实验的 `config.py`：数据根目录（`LOSO_BASE_DATA_DIR` 或 `DATA_ROOT`）、结果目录、`DEVICE`、目标受试者及训练轮数。原配置中的 `/hdd/...` 路径仅适用于原实验环境；如果机器只有一块 GPU，应把 `cuda:1` 改为 `cuda:0`，也可以使用 `cpu`。

例如，运行 DB2 融合实验：

```bash
python DB2/TEST_DB2/main_fusion_finetune.py
```

其他数据集可将路径替换为 `DB5/TEST_DB5/`、`DB7/TEST_DB7/` 或 `EPN/TEST_EPN/`；运行基线时则替换为相应的 `baseline_*` 目录。脚本会在配置的输出目录写入训练产物，并在实验目录更新汇总 CSV。完整训练需要预先下载的数据集和相应计算资源。

## 实验划分提示

- DB2/DB5/DB7 的 `TEST_*` 配置默认使用跨受试者预训练，并在目标受试者上按重复次数划分微调与测试数据；具体受试者、动作集及重复编号以各自 `config.py` 为准。
- EPN 的 `TEST_EPN` 配置限定 200 Hz sEMG 用户，以 `training` 组内的重复 1–3 微调、4–15 测试；代码只分类 11 个有效手势，不把静息类计入 `NUM_CLASSES`。
- 各目录配置和结果文件属于独立实验，请避免直接比较不同数据集上的准确率。
