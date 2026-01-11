
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs, rink

def draw_rink(ax):
    ax.add_patch(Rectangle((-100, -42.5), 200, 85, fill=False, edgecolor='black', linewidth=1, zorder=0))
    ax.axvline(0, color='red', linewidth=1, alpha=0.3)
    ax.axvline(25, color='blue', linewidth=1, alpha=0.3)
    ax.axvline(89, color='red', linewidth=1, alpha=0.3)
    ax.add_patch(Circle((89, 0), 6, color='lightblue', alpha=0.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_aspect('equal')

def main():
    print("--- Comparing All Empirical Data (Unfiltered) ---")
    summary_path = 'analysis/blocked_shots/blocked_shots_summary_batch.csv'
    df_sum = pd.read_csv(summary_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for col, role in enumerate(['F', 'D']):
        # Row 1: Empirical (Score > 0.4) - The previous benchmark
        ax = axes[0, col]
        draw_rink(ax)
        subset_high = df_sum[(df_sum['shooter_role'] == role) & (df_sum['score'] > 0.4)].copy()
        tx = np.where(subset_high['x'] < 0, -subset_high['x'], subset_high['x'])
        ty = np.where(subset_high['x'] < 0, -subset_high['y'], subset_high['y'])
        sns.kdeplot(x=tx, y=ty, cmap='Greens', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Empirical (Score > 0.4, N={len(subset_high)})")
        
        # Row 2: Empirical (All Tracked) - EVERYTHING
        ax = axes[1, col]
        draw_rink(ax)
        subset_all = df_sum[df_sum['shooter_role'] == role].copy()
        ax_all = np.where(subset_all['x'] < 0, -subset_all['x'], subset_all['x'])
        ay_all = np.where(subset_all['x'] < 0, -subset_all['y'], subset_all['y'])
        sns.kdeplot(x=ax_all, y=ay_all, cmap='Oranges', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Empirical (All Tracked, N={len(subset_all)})")

    plt.tight_layout()
    out_path = 'analysis/nested_xgs/empirical_unfiltered_comparison.png'
    plt.savefig(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
