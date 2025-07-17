import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

# Generate dummy data
np.random.seed(42)
num_points = 50
flops = np.random.uniform(10, 1000, num_points)
activation_size = np.random.uniform(20, 400, num_points)
latency = np.random.uniform(1, 50, num_points)

data = pd.DataFrame({
    'FLOPs': flops,
    'Activation_Size': activation_size,
    'Latency': latency
})

# Create subplots for different visualization methods
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Different Ways to Show 3 Variables in 2D Plots', fontsize=16, fontweight='bold')

# Method 1: Color + Size (like your original)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(data['FLOPs'], data['Activation_Size'], 
                      c=data['Latency'], s=data['Latency']*8, 
                      cmap='viridis', alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('FLOPs (GFLOPs)')
ax1.set_ylabel('Activation Size (MB)')
ax1.set_title('Method 1: Color + Size')
plt.colorbar(scatter1, ax=ax1, label='Latency (ms)')

# Method 2: Color + Shape (using different markers)
ax2 = axes[0, 1]
# Bin latency into categories for different shapes
latency_bins = pd.cut(data['Latency'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
markers = ['o', 's', '^', 'D']
colors = ['blue', 'green', 'orange', 'red']

for i, (category, marker, color) in enumerate(zip(['Low', 'Medium', 'High', 'Very High'], markers, colors)):
    mask = latency_bins == category
    ax2.scatter(data['FLOPs'][mask], data['Activation_Size'][mask], 
               marker=marker, c=color, s=60, alpha=0.7, 
               label=f'{category} Latency', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('FLOPs (GFLOPs)')
ax2.set_ylabel('Activation Size (MB)')
ax2.set_title('Method 2: Color + Shape')
ax2.legend()

# Method 3: Contour Plot with Scatter
ax3 = axes[0, 2]
# Create a grid for contour plotting
xi = np.linspace(data['FLOPs'].min(), data['FLOPs'].max(), 50)
yi = np.linspace(data['Activation_Size'].min(), data['Activation_Size'].max(), 50)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate latency values onto the grid
from scipy.interpolate import griddata
Zi = griddata((data['FLOPs'], data['Activation_Size']), data['Latency'], 
              (Xi, Yi), method='cubic', fill_value=data['Latency'].mean())

contour = ax3.contour(Xi, Yi, Zi, levels=8, colors='gray', alpha=0.5, linewidths=1)
ax3.clabel(contour, inline=True, fontsize=8)
scatter3 = ax3.scatter(data['FLOPs'], data['Activation_Size'], 
                      c=data['Latency'], cmap='plasma', s=40, 
                      edgecolor='black', linewidth=0.5)
ax3.set_xlabel('FLOPs (GFLOPs)')
ax3.set_ylabel('Activation Size (MB)')
ax3.set_title('Method 3: Contour + Scatter')
plt.colorbar(scatter3, ax=ax3, label='Latency (ms)')

# Method 4: Bubble Chart with Transparency
ax4 = axes[1, 0]
# Normalize latency for alpha values
alpha_values = (data['Latency'] - data['Latency'].min()) / (data['Latency'].max() - data['Latency'].min())
alpha_values = 0.3 + 0.7 * alpha_values  # Scale to 0.3-1.0 range

for i in range(len(data)):
    ax4.scatter(data['FLOPs'].iloc[i], data['Activation_Size'].iloc[i], 
               s=200, c='red', alpha=alpha_values.iloc[i], 
               edgecolor='black', linewidth=1)

ax4.set_xlabel('FLOPs (GFLOPs)')
ax4.set_ylabel('Activation Size (MB)')
ax4.set_title('Method 4: Size + Transparency')

# Method 5: Arrow Plot (showing direction/magnitude)
ax5 = axes[1, 1]
# Use latency to determine arrow length and color
max_latency = data['Latency'].max()
for i in range(0, len(data), 3):  # Show every 3rd point to avoid clutter
    x, y, lat = data['FLOPs'].iloc[i], data['Activation_Size'].iloc[i], data['Latency'].iloc[i]
    # Arrow length proportional to latency
    arrow_length = (lat / max_latency) * 50
    ax5.arrow(x, y, arrow_length, 0, head_width=10, head_length=15, 
             fc=plt.cm.coolwarm(lat/max_latency), ec='black', alpha=0.7)

ax5.set_xlabel('FLOPs (GFLOPs)')
ax5.set_ylabel('Activation Size (MB)')
ax5.set_title('Method 5: Arrow Length + Color')

# Method 6: Text Annotations
ax6 = axes[1, 2]
scatter6 = ax6.scatter(data['FLOPs'], data['Activation_Size'], 
                      c='lightblue', s=100, alpha=0.6, 
                      edgecolor='black', linewidth=1)

# Add text annotations for latency values (show every 5th point)
for i in range(0, len(data), 5):
    ax6.annotate(f'{data["Latency"].iloc[i]:.1f}', 
                (data['FLOPs'].iloc[i], data['Activation_Size'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

ax6.set_xlabel('FLOPs (GFLOPs)')
ax6.set_ylabel('Activation Size (MB)')
ax6.set_title('Method 6: Text Annotations')

plt.tight_layout()
plt.show()

# Alternative: Single best method - Bubble chart with color and size
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['FLOPs'], data['Activation_Size'], 
                     c=data['Latency'], s=data['Latency']*15, 
                     cmap='viridis', alpha=0.7, edgecolor='black', linewidth=1)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Latency (ms)', fontsize=12)

# Customize plot
plt.xlabel('FLOPs (GFLOPs)', fontsize=12, fontweight='bold')
plt.ylabel('Activation Size (MB)', fontsize=12, fontweight='bold')
plt.title('Neural Network Performance: FLOPs vs Activation Size\n(Bubble size and color represent Latency)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add legend for bubble sizes
sizes = [10, 25, 40]
labels = ['Low Latency', 'Medium Latency', 'High Latency']
legend_elements = [plt.scatter([], [], s=s*15, c='gray', alpha=0.7, edgecolor='black') 
                  for s in sizes]
plt.legend(legend_elements, labels, scatterpoints=1, loc='upper left', 
          title='Latency Levels', title_fontsize=10)

plt.tight_layout()
plt.show()
