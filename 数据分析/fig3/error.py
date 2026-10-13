import numpy as np
import matplotlib.pyplot as plt

# 加载数据
true_data = np.load('true-1.npy')
linear_data = np.load('linear-1.npy')
mlp_data = np.load('mlp-1.npy')
sp9_data = np.load('sp9-1.npy')

# 物理尺寸 (mm)
L, H = 295, 48.835

# 创建图形：2行4列
fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)

data_list = [linear_data, mlp_data, sp9_data, true_data]
titles = ['Linear', 'MLP', 'SP9', 'True']
errors = [linear_data - true_data, mlp_data - true_data, sp9_data - true_data, None]

# 全局colorbar范围
vmin = min(d.min() for d in data_list)
vmax = max(d.max() for d in data_list)
err_vals = [abs(e.min()) for e in errors[:3]] + [abs(e.max()) for e in errors[:3]]
err_max = max(err_vals)

for i, (ax_col, d, title, err) in enumerate(zip(axes.T, data_list, titles, errors)):
    # 上行：原温度场
    im1 = ax_col[0].imshow(d, aspect='auto', origin='lower', vmin=vmin, vmax=vmax,
                            extent=[0, L, 0, H])
    ax_col[0].set_ylabel('y (mm)')
    ax_col[0].set_title(title)
    ax_col[0].set_xlabel('x (mm)')

    # 下行：误差（前三列）
    if err is not None:
        im2 = ax_col[1].imshow(err, aspect='auto', origin='lower', cmap='RdBu_r',
                                vmin=-err_max, vmax=err_max, extent=[0, L, 0, H])
        ax_col[1].set_xlabel('x (mm)')
        ax_col[1].set_title(f'{title} Error')
    else:
        ax_col[1].axis('off')

# 添加共享colorbar
fig.colorbar(im1, ax=axes[0,:], label='Temperature', shrink=0.8, location='right')
fig.colorbar(im2, ax=axes[1,:3], label='Error', shrink=0.8, location='right')

plt.savefig('temperature_with_error.png', dpi=150, bbox_inches='tight')
print('已保存为 temperature_with_error.png')