# sEMG-IMU Gesture Recognition

基于表面肌电（sEMG）与惯性传感器（IMU）信号的手势识别实验代码，涵盖 NinaPro DB2、DB5、DB7 和 EPN-107 数据集。仓库包含单模态基线、融合模型，以及跨受试者预训练与微调实验。

## 目录

- `DB2/`、`DB5/`、`DB7/`：NinaPro 数据集实验。
- `EPN/`：EPN-107 数据集实验。
- 每个数据集下的 `baseline_*` 为融合基线，`baseline_*_IMU` 和 `baseline_*_SEMG` 为单模态基线，`TEST_*` 为融合模型实验；目录中的 CSV 为实验结果摘要。

## 运行

准备对应数据集及 Python 环境，按需修改实验目录中的 `config.py`（尤其是数据路径、输出路径与 GPU 设备），然后运行相应的 `main_fusion_finetune.py`，例如：

```bash
python DB2/TEST_DB2/main_fusion_finetune.py
```

数据集与训练输出未包含在本仓库中。
