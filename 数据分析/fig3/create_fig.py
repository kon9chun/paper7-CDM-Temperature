import numpy as np
import matplotlib.pyplot as plt

# Load data
sp9_data = np.load('ydata.npy')
print("sp9 shape:", sp9_data.shape)

# Data is 2D with shape (40, 240) representing the spatial field
data = sp9_data[50]

# Red point positions
sp9_positions = np.array([[0,0],[0,120],[0,239],[19,0],[19,120],[19,239],[39,0],[39,120],[39,239]])
sp5_positions = np.array([[0,0],[0,239],[19,120],[39,0],[39,239]])
sp3_positions = np.array([[0,0],[19,120],[39,239]])

# Convert indices to physical coordinates (meters)
def idx_to_m(pos, x_max=0.295, y_max=0.048835):
    x = pos[:, 1] * x_max / 239
    y = pos[:, 0] * y_max / 39
    return x, y

sp9_x, sp9_y = idx_to_m(sp9_positions)
sp5_x, sp5_y = idx_to_m(sp5_positions)
sp3_x, sp3_y = idx_to_m(sp3_positions)

# Figure dimensions: 295mm x 48.835mm
# For journal-quality, use appropriate DPI
# 295mm宽, 48.835mm高的子图，3个垂直排列
# 295mm = 11.614 inch, 48.835mm = 1.923 inch
fig, axes = plt.subplots(3, 1, figsize=(11.614, 5.77), dpi=300)

# Set physical limits
for ax in axes:
    ax.set_xlim(0, 0.295)
    ax.set_ylim(0, 0.048835)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal', adjustable='box')

# Create meshgrid
x = np.linspace(0, 0.295, 240)
y = np.linspace(0, 0.048835, 40)
X, Y = np.meshgrid(x, y)

cmap = 'viridis'

# sp9 subplot
axes[0].pcolormesh(X, Y, data, cmap=cmap, shading='auto')
axes[0].scatter(sp9_x, sp9_y, c='red', s=60, marker='o', edgecolors='black', linewidths=0.8, zorder=5)
axes[0].set_title('The sampling point location of SP9', fontsize=11, fontweight='bold')

# sp5 subplot
axes[1].pcolormesh(X, Y, data, cmap=cmap, shading='auto')
axes[1].scatter(sp5_x, sp5_y, c='red', s=60, marker='o', edgecolors='black', linewidths=0.8, zorder=5)
axes[1].set_title('The sampling point location of SP5', fontsize=11, fontweight='bold')

# sp3 subplot
axes[2].pcolormesh(X, Y, data, cmap=cmap, shading='auto')
axes[2].scatter(sp3_x, sp3_y, c='red', s=60, marker='o', edgecolors='black', linewidths=0.8, zorder=5)
axes[2].set_title('The sampling point location of SP3', fontsize=11, fontweight='bold')

# Clean layout without colorbar to match reference style
plt.tight_layout()
plt.savefig('result.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('result.pdf', bbox_inches='tight', facecolor='white')
print("Figures saved as result.png and result.pdf")