# High-Resolution Reconstruction of Lithium-Ion Battery Internal Temperature Fields Using Conditional Diffusion Models

Code and data release for Paper 7 (CDM-based battery internal temperature field reconstruction).

电池内部温度场高分辨率重建 —— 条件扩散模型（CDM / DDPM）方法代码与数据。

## Overview | 项目简介

Reconstruct the full 2-D internal temperature field (40 × 240) of a LiFePO4 blade battery
(182.5 Ah) from a small number of sparse surface temperature samples using a conditional
diffusion model (U-Net denoiser + time embedding + condition embedding).

- Sampling configurations: 3 / 5 / 9 sensor points
- Baselines: linear interpolation, MLP regression
- Normalized RMSE down to 2.25e-4 (9 points), ~119x better than linear interpolation

## Repository Structure | 目录结构

```
.
├── diffusion_model.py        # UNet + DDPM conditional diffusion model (train/inference API)
├── traincc.py                # Training script, 9 sampling points (cond_dim=11)
├── trainccsp5.py             # Training script, 5 sampling points (cond_dim=7)
├── trainccsp3.py             # Training script, 3 sampling points (cond_dim=5)
├── samplingsp9.py            # Sampling / evaluation script, 9 points
├── samplingsp5.py            # Sampling / evaluation script, 5 points
├── train.ipynb               # Training notebook
├── samplingsp3.ipynb         # Sampling notebook, 3 points
├── samplingsp5.ipynb         # Sampling notebook, 5 points
├── mlp-sp9.ipynb             # MLP baseline (9 points)
├── noise_sp9.ipynb           # Sensor-noise robustness experiments
├── valdata.ipynb             # Validation data preparation
├── sp.npy                    # Sparse sampled points, shape (N, 9)
├── embdata.npy               # Condition vectors (9 points), shape (N, 11)
├── *_rmse.npy                # Per-sample RMSE results (sp3/sp5/sp9/noise/mlp)
├── linear/                   # Linear-interpolation baseline & RMSE analysis
├── 数据分析/fig1~fig3/        # Analysis scripts and figures (data-analysis)
├── final_model.pt            # CDM checkpoint, 9 sampling points
├── final_model_sp5_0319.pt   # CDM checkpoint, 5 sampling points
├── final_model_sp3_0320.pt   # CDM checkpoint, 3 sampling points
└── mlp_sp9.pth               # MLP baseline checkpoint
                              # (all tracked with Git LFS, see below)
```

## Requirements | 环境依赖

```bash
pip install -r requirements.txt
```

Python >= 3.9, PyTorch >= 2.0 (CUDA recommended for training; inference works on CPU).

## Data Format | 数据格式

| File | Shape | Description |
|---|---|---|
| `ydata.npy` (not included, see below) | (N, 40, 240) | Ground-truth internal temperature fields (normalized to [-1, 1]) |
| `sp.npy` | (N, 9) | Sparse temperature samples (9 sensor points per sample) |
| `embdata.npy` | (N, 11) | Condition vector: 9 normalized samples + normalized max & min |

Normalization follows `samplingsp9.py`: Min-Max to [-1, 1] per sample, using the
per-sample sampled-point max/min as the scaling range.

The training set uses 83,513 samples and the validation set 10,000 samples
(random split, seed 42), generated from a P2D + 3-D thermal coupled simulation of a
182.5 Ah LiFePO4 blade battery across 3 ambient temperatures (288/298/308 K) and
9 driving cycles (0.5C/1C/2C charge & discharge, NEDC/UDDS/WLTC).

## Usage | 使用说明

Training (9 points):

```bash
python traincc.py        # saves final_model.pt
```

Training (5 / 3 points):

```bash
python trainccsp5.py     # saves final_model_sp5.pt
python trainccsp3.py     # saves final_model_sp3.pt
```

Evaluation:

```bash
python samplingsp9.py    # loads final_model.pt, saves sp9_rmse.npy / sp9-1.npy / true-1.npy
python samplingsp5.py    # loads final_model_sp5.pt, saves sp5_rmse.npy / sp5_pred.npy
```

Model checkpoints (Git LFS):

```bash
git lfs install
git lfs pull
```

## Large Files | 大文件说明

- Model checkpoints (`final_model.pt`, `final_model_sp5_0319.pt`, `final_model_sp3_0320.pt`,
  `mlp_sp9.pth`, ~0.85 GB total) are tracked with **Git LFS**. Run `git lfs pull` after
  cloning to download them.
- The full dataset `ydata.npy` (~3.4 GB) and prediction arrays (`sp5_pred.npy`, etc.)
  exceed GitHub's 100 MB per-file limit and are **not included**. They are available
  from the authors on reasonable request.

## Citation | 引用

```bibtex
@article{cdm_battery_temperature,
  title  = {High-Resolution Reconstruction of Lithium-Ion Battery Internal
            Temperature Fields Using Conditional Diffusion Models},
  author = {Tsinghua University, School of Vehicle and Mobility},
  year   = {2026}
}
```

## License | 许可

MIT License
