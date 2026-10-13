import numpy as np
import matplotlib.pyplot as plt

linear = np.load('linear_rmse.npy')
mlp = np.load('mlp_rmse.npy')
sp9 = np.load('sp9_rmse.npy')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. Box plot (log scale)
ax1 = axes[0, 0]
bp = ax1.boxplot([linear, mlp, sp9], tick_labels=['Linear', 'MLP', 'SP9(CDM)'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('RMSE')
ax1.set_title('(a) Box Plot (Log Scale)')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 2. Histogram with log y-axis
ax2 = axes[0, 1]
bins = np.logspace(-5, 0, 50)
ax2.hist(linear, bins=bins, alpha=0.6, label='Linear', color='#FF6B6B', density=True)
ax2.hist(mlp, bins=bins, alpha=0.6, label='MLP', color='#4ECDC4', density=True)
ax2.hist(sp9, bins=bins, alpha=0.6, label='SP9(CDM)', color='#45B7D1', density=True)
ax2.set_xlabel('RMSE')
ax2.set_ylabel('Density (Log Scale)')
ax2.set_title('(b) Histogram (Log Scale)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Violin plot (log scale)
ax3 = axes[0, 2]
parts = ax3.violinplot([linear, mlp, sp9], positions=[1, 2, 3], showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(['Linear', 'MLP', 'SP9(CDM)'])
ax3.set_ylabel('RMSE')
ax3.set_yscale('log')
ax3.set_title('(c) Violin Plot (Log Scale)')
ax3.grid(True, alpha=0.3)

# 4. CDF (log x-axis)
ax4 = axes[1, 0]
for data, label, color in zip([linear, mlp, sp9], ['Linear', 'MLP', 'SP9(CDM)'], colors):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.plot(sorted_data, cdf, label=label, color=color, linewidth=2)
ax4.set_xlabel('RMSE (Log Scale)')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('(d) CDF Comparison')
ax4.set_xscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Q-Q plot: MLP vs SP9 (CDM)
ax5 = axes[1, 1]
quantiles = np.linspace(0, 1, 100)
sp9_quantiles = np.quantile(sp9, quantiles)
mlp_quantiles = np.quantile(mlp, quantiles)
linear_quantiles = np.quantile(linear, quantiles)
ax5.scatter(sp9_quantiles, mlp_quantiles, alpha=0.5, s=10, label='MLP vs SP9', color='#4ECDC4')
ax5.scatter(sp9_quantiles, linear_quantiles, alpha=0.5, s=10, label='Linear vs SP9', color='#FF6B6B')
max_val = max(linear_quantiles.max(), mlp_quantiles.max())
ax5.plot([0, max_val], [0, max_val], 'k--', label='y=x')
ax5.set_xlabel('SP9 (CDM) Quantiles')
ax5.set_ylabel('Model Quantiles')
ax5.set_title('(e) Q-Q Plot (vs SP9 Reference)')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Error reduction bar chart (log scale)
ax6 = axes[1, 2]
models = ['Linear', 'MLP', 'SP9(CDM)']
means = [linear.mean(), mlp.mean(), sp9.mean()]
bars = ax6.bar(models, means, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Mean RMSE')
ax6.set_title('(f) Mean RMSE Comparison (Log Scale)')
ax6.set_yscale('log')
for bar, mean in zip(bars, means):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, f'{mean:.6f}',
             ha='center', va='bottom', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Updated visualization saved!")
