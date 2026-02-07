"""
Ablation Report — Visual-Only vs Multimodal Bar Chart
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================= DATA =========================
models = ['Visual Only', 'Visual + rPPG\n(Multimodal)']
binary_acc = [82.3, 87.6]
multi_acc = [47.5, 53.4]

x = np.arange(len(models))
width = 0.3

# ========================= PLOT =========================
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, binary_acc, width, label='Binary (Task 5a)',
               color='#2196F3', edgecolor='white', linewidth=1.5, zorder=3)
bars2 = ax.bar(x + width/2, multi_acc, width, label='Multi-class (Task 5b)',
               color='#FF9800', edgecolor='white', linewidth=1.5, zorder=3)

# Value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=13, color='#1565C0')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=13, color='#E65100')

# Improvement arrow for multi-class
ax.annotate('', xy=(1 + width/2, 53.4), xytext=(0 + width/2, 47.5),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5))
ax.text(0.75, 50, '+5.9%', fontsize=14, fontweight='bold', color='#2E7D32',
        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', 
                                edgecolor='#4CAF50', alpha=0.9))

# Styling
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Ablation Study: Impact of rPPG Features on Engagement Prediction',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Highlight box
ax.text(0.98, 0.02, 'rPPG features improve\nmulti-class by +5.9%\nbinary by +5.3%',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        color='#2E7D32', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))

plt.tight_layout()

os.makedirs('graphs', exist_ok=True)
out_path = 'graphs/ablation_report.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
