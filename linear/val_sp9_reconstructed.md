# val_sp9_reconstructed.npy 生成说明

## 1. 数据概述

- **输入文件**：`val_sp9.npy`
- **输入形状**：(10000, 9) — 10000 个样本，每个样本 9 个数据点
- **输出文件**：`val_sp9_reconstructed.npy`
- **输出形状**：(10000, 40, 240) — 10000 个 40×240 的重建矩阵

## 2. 9 个已知点的位置

9 个点分布在 3 行、3 列的稀疏网格上：

| 行 \ 列 | col=0 | col=120 | col=239 |
|---------|-------|---------|---------|
| **row=0** | P0 [0,0] | P1 [0,120] | P2 [0,239] |
| **row=19** | P3 [19,0] | P4 [19,120] | P5 [19,239] |
| **row=39** | P6 [39,0] | P7 [39,120] | P8 [39,239] |

对应 `val_sp9.npy` 中每行的 9 个值按顺序为：P0, P1, P2, P3, P4, P5, P6, P7, P8

## 3. 线性插值原理

采用**双线性插值**（Bilinear Interpolation），分两步进行：

### 步骤 1：列方向插值（生成 3 条完整行）

对 row=0、row=19、row=39 这三行，分别利用左右两个锚点进行线性插值，填充所有 240 列：

```
row0[col] = P0 + (P1 - P0) * col / 120          (col ∈ [0,120])
row0[col] = P1 + (P2 - P1) * (col-120) / 119   (col ∈ [120,239])
```

row=19 和 row=39 同理。

### 步骤 2：行方向插值（填充所有行）

对于每一列 c，先在 row=0 和 row=19 之间插值（填充 row 1~18）：

```
row[r, c] = row0[c] + (row19[c] - row0[c]) * r / 19    (r ∈ [1,18])
```

再在 row=19 和 row=39 之间插值（填充 row 20~38）：

```
row[r, c] = row19[c] + (row39[c] - row19[c]) * (r-19) / 20    (r ∈ [20,38])
```

row=0、row=19、row=39 保持原始值不变。

## 4. 完整代码

```python
import numpy as np

# 加载数据
data = np.load('val_sp9.npy')
# data.shape = (10000, 9)

def interpolate_row(val0, val1, col0, col1, target_col):
    """一维线性插值：沿列方向"""
    if col1 == col0:
        return val0
    t = (target_col - col0) / (col1 - col0)
    return val0 + t * (val1 - val0)

def interpolate_2d(sample):
    """
    双线性插值：将 9 点稀疏数据还原为 40x240 矩阵
    sample: 长度为 9 的数组，按顺序对应 9 个已知点的值
    """
    output = np.zeros((40, 240), dtype=np.float32)

    # ---- 步骤 1：列方向插值，生成 3 条完整行 ----
    # row = 0：利用 P0-P1 (col 0-120) 和 P1-P2 (col 120-239)
    row0 = np.array([
        interpolate_row(sample[0], sample[1], 0, 120, c) if c <= 120
        else interpolate_row(sample[1], sample[2], 120, 239, c)
        for c in range(240)
    ])

    # row = 19：利用 P3-P4 和 P4-P5
    row19 = np.array([
        interpolate_row(sample[3], sample[4], 0, 120, c) if c <= 120
        else interpolate_row(sample[4], sample[5], 120, 239, c)
        for c in range(240)
    ])

    # row = 39：利用 P6-P7 和 P7-P8
    row39 = np.array([
        interpolate_row(sample[6], sample[7], 0, 120, c) if c <= 120
        else interpolate_row(sample[7], sample[8], 120, 239, c)
        for c in range(240)
    ])

    # ---- 步骤 2：行方向插值，填充所有行 ----
    for c in range(240):
        # 已知行保持原值
        output[0, c]  = row0[c]
        output[19, c] = row19[c]
        output[39, c] = row39[c]

        # row 1~18：在 row0 和 row19 之间插值
        for r in range(1, 19):
            t = r / 19
            output[r, c] = row0[c] + t * (row19[c] - row0[c])

        # row 20~38：在 row19 和 row39 之间插值
        for r in range(20, 39):
            t = (r - 19) / 20
            output[r, c] = row19[c] + t * (row39[c] - row19[c])

    return output

# 对所有 10000 个样本进行处理
output = np.array([interpolate_2d(data[i]) for i in range(data.shape[0])])
# output.shape = (10000, 40, 240)

# 保存结果
np.save('val_sp9_reconstructed.npy', output)
```

## 5. 插值验证

以样本 0 为例，所有 9 个已知点精确还原：

| 位置 | 输入值 | 输出值 |
|------|--------|--------|
| [0,0] | 304.4416 | 304.4416 |
| [0,120] | 304.1542 | 304.1542 |
| [0,239] | 303.3066 | 303.3066 |
| [19,0] | 304.5230 | 304.5230 |
| [19,120] | 304.2527 | 304.2527 |
| [19,239] | 303.4556 | 303.4556 |
| [39,0] | 304.5514 | 304.5514 |
| [39,120] | 304.2899 | 304.2899 |
| [39,239] | 303.5190 | 303.5190 |

插值点验证（如 [10, 60]）计算值与输出值一致。

## 6. 文件对应关系

```
val_sp9.npy                  val_sp9_reconstructed.npy
┌─────────────────────┐      ┌─────────────────────────┐
│ (10000, 9)           │      │ (10000, 40, 240)        │
│ 每行9个稀疏点值       │ ──→  │ 每行40×240重建矩阵       │
└─────────────────────┘      └─────────────────────────┘
```
