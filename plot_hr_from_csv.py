"""
=========================================================
 rPPG: Heart Rate vs Video Duration (from summary.csv)
 
 Data Source: task3_rppg/results/summary.csv
 No video processing — purely CSV-based graphs.
 
 Usage:
   python plot_hr_from_csv.py
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ========================= LOAD DATA =========================
CSV_PATH = 'task3_rppg/results/summary.csv'
df = pd.read_csv(CSV_PATH)

# Compute duration (seconds) from frames and fps
df['duration_sec'] = df['frames'] / df['fps']

# Handle DeepPhys column (may not exist in old CSVs)
if 'hr_deepphys' not in df.columns:
    df['hr_deepphys'] = 0.0
    df['sqi_deepphys'] = 0.0

# Valid only (HR > 0)
df_valid = df[df['hr_pos'] > 0].copy()
df_corrupt = df[df['hr_pos'] == 0].copy()

# Plausible HR flag
df_valid['pos_ok'] = (df_valid['hr_pos'] >= 40) & (df_valid['hr_pos'] <= 120)
df_valid['chrom_ok'] = (df_valid['hr_chrom'] >= 40) & (df_valid['hr_chrom'] <= 120)
df_valid['deep_ok'] = (df_valid['hr_deepphys'] >= 40) & (df_valid['hr_deepphys'] <= 120)

# Short video name
df_valid['short'] = df_valid['video'].str.replace('subject_', 'S').str.replace('_Vid_', 'V')

os.makedirs('graphs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"Loaded {len(df)} videos from {CSV_PATH}")
print(f"Valid: {len(df_valid)} | Corrupt: {len(df_corrupt)}")

# ================================================================
# GRAPH 1: Heart Rate vs Video Duration (Scatter)
# ================================================================
print("\n[1/3] HR vs Duration scatter...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# POS
ok = df_valid[df_valid['pos_ok']]
bad = df_valid[~df_valid['pos_ok']]

axes[0].scatter(ok['duration_sec'], ok['hr_pos'], c='#2196F3', s=60, alpha=0.7,
               edgecolors='white', linewidth=0.5, label=f'Plausible ({len(ok)})', zorder=3)
axes[0].scatter(bad['duration_sec'], bad['hr_pos'], c='#F44336', s=60, alpha=0.7,
               marker='^', edgecolors='white', linewidth=0.5, label=f'Outlier ({len(bad)})', zorder=3)

# Annotate outliers
for _, row in bad.iterrows():
    axes[0].annotate(row['short'], (row['duration_sec'], row['hr_pos']),
                    fontsize=7, color='#F44336', xytext=(5, 5), textcoords='offset points')

axes[0].axhspan(60, 100, alpha=0.06, color='#4CAF50', label='Normal resting (60-100)')
axes[0].axhline(y=np.mean(ok['hr_pos']), color='#2196F3', linestyle=':', alpha=0.5,
               label=f'Mean: {np.mean(ok["hr_pos"]):.1f} BPM')
axes[0].set_xlabel('Video Duration (seconds)')
axes[0].set_ylabel('Heart Rate (BPM)')
axes[0].set_title('POS Algorithm', fontweight='bold')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].set_ylim(30, 170)

# CHROM
ok_c = df_valid[df_valid['chrom_ok']]
bad_c = df_valid[~df_valid['chrom_ok']]

axes[1].scatter(ok_c['duration_sec'], ok_c['hr_chrom'], c='#FF9800', s=60, alpha=0.7,
               edgecolors='white', linewidth=0.5, label=f'Plausible ({len(ok_c)})', zorder=3)
axes[1].scatter(bad_c['duration_sec'], bad_c['hr_chrom'], c='#F44336', s=60, alpha=0.7,
               marker='^', edgecolors='white', linewidth=0.5, label=f'Outlier ({len(bad_c)})', zorder=3)

for _, row in bad_c.iterrows():
    axes[1].annotate(row['short'], (row['duration_sec'], row['hr_chrom']),
                    fontsize=7, color='#F44336', xytext=(5, 5), textcoords='offset points')

axes[1].axhspan(60, 100, alpha=0.06, color='#4CAF50', label='Normal resting (60-100)')
axes[1].axhline(y=np.mean(ok_c['hr_chrom']), color='#FF9800', linestyle=':', alpha=0.5,
               label=f'Mean: {np.mean(ok_c["hr_chrom"]):.1f} BPM')
axes[1].set_xlabel('Video Duration (seconds)')
axes[1].set_ylabel('Heart Rate (BPM)')
axes[1].set_title('CHROM Algorithm', fontweight='bold')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].set_ylim(30, 170)

# Watermark
fig.text(0.5, -0.02, f'Source: {CSV_PATH} | Generated: {timestamp} | '
         f'Reproduce: python plot_hr_from_csv.py',
         ha='center', fontsize=7, color='gray', fontfamily='monospace')

plt.suptitle('Heart Rate vs Video Duration\n(Each point = 1 video from summary.csv)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/rppg_hr_vs_duration.png')
plt.close()
print("  Saved: graphs/rppg_hr_vs_duration.png")

# ================================================================
# GRAPH 2: HR vs Subject (Timeline-style)
# ================================================================
print("[2/3] HR vs Subject timeline...")

fig, ax = plt.subplots(figsize=(18, 7))

df_sorted = df_valid.sort_values('video').reset_index(drop=True)
x = np.arange(len(df_sorted))

# Plot both algorithms
ax.plot(x, df_sorted['hr_pos'], 'o-', color='#2196F3', linewidth=1.5,
        markersize=5, alpha=0.8, label='POS HR')
ax.plot(x, df_sorted['hr_chrom'], 's-', color='#FF9800', linewidth=1.5,
        markersize=5, alpha=0.8, label='CHROM HR')

# Fill between to show difference
ax.fill_between(x, df_sorted['hr_pos'], df_sorted['hr_chrom'], alpha=0.1, color='#9C27B0')

# Normal range band
ax.axhspan(60, 100, alpha=0.06, color='#4CAF50', label='Normal resting (60-100)')
ax.axhline(y=120, color='#F44336', linestyle='--', alpha=0.4, label='Plausible upper (120)')
ax.axhline(y=40, color='#F44336', linestyle='--', alpha=0.4, label='Plausible lower (40)')

# Mean lines
ax.axhline(y=df_sorted['hr_pos'].mean(), color='#2196F3', linestyle=':', alpha=0.4)
ax.axhline(y=df_sorted['hr_chrom'].mean(), color='#FF9800', linestyle=':', alpha=0.4)

# X labels (every 3rd video)
short_names = df_sorted['short'].values
ax.set_xticks(x[::3])
ax.set_xticklabels(short_names[::3], rotation=55, ha='right', fontsize=7)

# Mark outliers
for i, row in df_sorted.iterrows():
    if row['hr_pos'] > 120 or row['hr_chrom'] > 120:
        ax.annotate(f'{row["short"]}\n{max(row["hr_pos"], row["hr_chrom"]):.0f}',
                   (i, max(row['hr_pos'], row['hr_chrom'])),
                   fontsize=7, color='#F44336', fontweight='bold',
                   xytext=(0, 10), textcoords='offset points', ha='center')

ax.set_xlabel('Video (sorted by name)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Heart Rate Across All {len(df_sorted)} Videos — POS vs CHROM\n'
             f'POS Mean: {df_sorted["hr_pos"].mean():.1f} BPM | '
             f'CHROM Mean: {df_sorted["hr_chrom"].mean():.1f} BPM',
             fontweight='bold', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-1, len(df_sorted))
ax.set_ylim(30, max(df_sorted[['hr_pos', 'hr_chrom']].max()) + 15)
ax.grid(True, alpha=0.3)

fig.text(0.5, -0.02, f'Source: {CSV_PATH} | Generated: {timestamp}',
         ha='center', fontsize=7, color='gray', fontfamily='monospace')

plt.tight_layout()
plt.savefig('graphs/rppg_hr_vs_subject.png')
plt.close()
print("  Saved: graphs/rppg_hr_vs_subject.png")

# ================================================================
# GRAPH 3: HR + SQI Combined (Dual Y-axis)
# ================================================================
print("[3/3] HR + SQI combined plot...")

fig, ax1 = plt.subplots(figsize=(18, 6))

df_s = df_valid.sort_values('video').reset_index(drop=True)
x = np.arange(len(df_s))

# Left Y: Heart Rate
color_hr = '#2196F3'
ax1.bar(x - 0.2, df_s['hr_chrom'], 0.4, color='#FF9800', alpha=0.7, label='CHROM HR', edgecolor='white')
ax1.bar(x + 0.2, df_s['hr_pos'], 0.4, color='#2196F3', alpha=0.7, label='POS HR', edgecolor='white')
ax1.axhline(y=120, color='#F44336', linestyle='--', alpha=0.3)
ax1.axhline(y=40, color='#F44336', linestyle='--', alpha=0.3)
ax1.set_ylabel('Heart Rate (BPM)', fontsize=11)
ax1.set_ylim(0, max(df_s[['hr_pos', 'hr_chrom']].max()) + 20)
ax1.legend(loc='upper left', fontsize=9)

# Right Y: SQI
ax2 = ax1.twinx()
ax2.plot(x, df_s['sqi_chrom'], 'D-', color='#4CAF50', markersize=4, linewidth=1.2,
         alpha=0.8, label='CHROM SQI')
ax2.plot(x, df_s['sqi_pos'], 'v-', color='#9C27B0', markersize=4, linewidth=1.2,
         alpha=0.8, label='POS SQI')
ax2.set_ylabel('Signal Quality Index (SQI)', fontsize=11, color='#4CAF50')
ax2.set_ylim(0.90, 1.0)
ax2.legend(loc='upper right', fontsize=9)

short_names = df_s['short'].values
ax1.set_xticks(x[::4])
ax1.set_xticklabels(short_names[::4], rotation=55, ha='right', fontsize=7)
ax1.set_xlabel('Video')
ax1.grid(True, alpha=0.2)

plt.title(f'Heart Rate & Signal Quality per Video ({len(df_s)} videos)\n'
          f'(from summary.csv — bars = HR, lines = SQI)',
          fontweight='bold', fontsize=13)

fig.text(0.5, -0.02, f'Source: {CSV_PATH} | Generated: {timestamp}',
         ha='center', fontsize=7, color='gray', fontfamily='monospace')

plt.tight_layout()
plt.savefig('graphs/rppg_hr_sqi_combined.png')
plt.close()
print("  Saved: graphs/rppg_hr_sqi_combined.png")

# ================================================================
print(f"\nDone! 3 graphs saved to graphs/")
for f in sorted(os.listdir('graphs')):
    if f.startswith('rppg_hr_') and f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join('graphs', f)) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
