"""
RMSE 误差可视化脚本
需要在具有 matplotlib 的环境中运行
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
rmse = np.load('linear_rmse.npy')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RMSE Error Analysis: Linear Interpolation Reconstruction', fontsize=16, fontweight='bold')

# 1. Histogram
ax1 = axes[0, 0]
ax1.hist(rmse, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(rmse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={rmse.mean():.4f}')
ax1.axvline(np.median(rmse), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(rmse):.4f}')
ax1.set_xlabel('RMSE', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('RMSE Distribution (Histogram)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Cumulative distribution
ax2 = axes[0, 1]
sorted_rmse = np.sort(rmse)
cdf = np.arange(1, len(sorted_rmse) + 1) / len(sorted_rmse)
ax2.plot(sorted_rmse, cdf, linewidth=2, color='steelblue')
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='50% line')
ax2.axvline(np.median(rmse), color='orange', linestyle='--', linewidth=1.5, label=f'Median={np.median(rmse):.4f}')
ax2.set_xlabel('RMSE', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Box plot
ax3 = axes[1, 0]
bp = ax3.boxplot(rmse, vert=True, patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.7)
ax3.set_ylabel('RMSE', fontsize=12)
ax3.set_title('RMSE Box Plot', fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics text box
stats_text = f'Statistics:\nMean: {rmse.mean():.4f}\nStd: {rmse.std():.4f}\nMedian: {np.median(rmse):.4f}\nMin: {rmse.min():.4f}\nMax: {rmse.max():.4f}'
ax3.text(1.25, rmse.mean(), stats_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Log-scale histogram
ax4 = axes[1, 1]
ax4.hist(rmse, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax4.set_xlabel('RMSE (log scale)', fontsize=12)
ax4.set_ylabel('Frequency (log scale)', fontsize=12)
ax4.set_title('RMSE Distribution (Log Scale)', fontsize=14)
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3, axis='both')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('linear_rmse_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("可视化已保存至 linear_rmse_analysis.png")
