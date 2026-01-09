
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs, rink, impute, correction

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
    print("--- Generating Final Shot Heatmaps (3x2 Balanced Role Comparison) ---")
    
    # 1. Load Empirical Tracking Summary (The "Benchmark")
    summary_path = 'analysis/blocked_shots/blocked_shots_summary_batch.csv'
    df_sum = pd.read_csv(summary_path)
    # FILTER BY SCORE > 0.4 as requested for ground truth
    df_emp = df_sum[df_sum['score'] > 0.4].copy()
    print(f"Loaded {len(df_emp)} high-score empirical records.")
    
    # 2. Load PBP for Unblocked Baseline
    pbp_path = 'data/20232024/20232024_df.csv'
    df_pbp = pd.read_csv(pbp_path)
    df_unblocked = df_pbp[df_pbp['event'].isin(['shot','goal','missed-shot'])].copy()
    df_unblocked = fit_xgs.enrich_data_with_bios(df_unblocked)
    
    # 3. Plotting (3 Rows, 2 Cols)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    for col, role in enumerate(['F', 'D']):
        subset_emp = df_emp[df_emp['shooter_role'] == role].copy()
        
        # --- ROW 1: Imputed (Self-Consistency Test on Summary) ---
        # We RECONSTRUCT the block and then IMPUTE it.
        # This proves the mapping logic on the EXACT SAME set of shots.
        print(f"Propagating {role} Imputation...")
        
        # nx, ny reconstruction
        dx = 89.0 - subset_emp['x']
        dy = 0.0 - subset_emp['y']
        dist_to_net = np.sqrt(dx**2 + dy**2)
        ux = dx / dist_to_net
        uy = dy / dist_to_net
        subset_emp['nx_recon'] = subset_emp['x'] + ux * subset_emp['distance_to_blocker']
        subset_emp['ny_recon'] = subset_emp['y'] + uy * subset_emp['distance_to_blocker']
        
        temp_df = pd.DataFrame({
            'x': subset_emp['nx_recon'],
            'y': subset_emp['ny_recon'],
            'shooter_role': role,
            'event': 'blocked-shot'
        })
        imputed = impute.impute_blocked_shot_origins(temp_df, method='cdf_mapping')
        
        ax = axes[0, col]
        draw_rink(ax)
        ix = np.where(imputed['imputed_x'] < 0, -imputed['imputed_x'], imputed['imputed_x'])
        iy = np.where(imputed['imputed_x'] < 0, -imputed['imputed_y'], imputed['imputed_y'])
        sns.kdeplot(x=ix, y=iy, cmap='Blues', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - CDF Imputed (Consistency Test, N={len(imputed)})")

        # --- ROW 2: Empirical Truth (Score > 0.4) ---
        ax = axes[1, col]
        draw_rink(ax)
        tx = np.where(subset_emp['x'] < 0, -subset_emp['x'], subset_emp['x'])
        ty = np.where(subset_emp['x'] < 0, -subset_emp['y'], subset_emp['y'])
        sns.kdeplot(x=tx, y=ty, cmap='Greens', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Empirical Truth (Score > 0.4, N={len(subset_emp)})")
        
        # --- ROW 3: Unblocked (PBP Baseline) ---
        ax = axes[2, col]
        draw_rink(ax)
        sub_unb = df_unblocked[df_unblocked['shooter_role'] == role].copy()
        if len(sub_unb) > 10000: sub_unb = sub_unb.sample(10000)
        ux = np.where(sub_unb['x'] < 0, -sub_unb['x'], sub_unb['x'])
        uy = np.where(sub_unb['x'] < 0, -sub_unb['y'], sub_unb['y'])
        sns.kdeplot(x=ux, y=uy, cmap='Reds', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Unblocked (All PBP, N={len(sub_unb)})")

    plt.tight_layout()
    plt.savefig('analysis/nested_xgs/shot_heatmaps.png')
    print("Saved to analysis/nested_xgs/shot_heatmaps.png")

if __name__ == "__main__":
    main()
