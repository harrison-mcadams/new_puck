
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs, correction, rink, impute

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
    print("--- Generating Final Shot Heatmaps (3x2 All Data Comparison) ---")
    
    # 1. Load Data
    pbp_path = 'data/20232024/20232024_df.csv'
    print(f"Loading PBP {pbp_path}...")
    df_pbp = pd.read_csv(pbp_path)
    
    summary_path = 'analysis/blocked_shots/blocked_shots_summary_batch.csv'
    print(f"Loading Empirical {summary_path}...")
    df_emp_all = pd.read_csv(summary_path)
    # FILTER BY SCORE > 0.4 for Row 2
    df_emp = df_emp_all[df_emp_all['score'] > 0.4].copy()
    
    # 2. Prepare Subsets
    # Row 1: All PBP Blocks (Imputed)
    print("Preparing Imputed Blocks (PBP)...")
    df_bl = df_pbp[df_pbp['event'] == 'blocked-shot'].copy()
    df_bl = correction.fix_blocked_shot_attribution(df_bl)
    df_bl = fit_xgs.enrich_data_with_bios(df_bl)
    
    # SHOOTER ROLE PROXY LOGIC:
    # In PBP, the ID belongs to the BLOCKER.
    # If Blocker = D, then Shooter = Likely F.
    # If Blocker = F, then Shooter = Likely D.
    df_bl['est_shooter_role'] = np.where(df_bl['shooter_role'] == 'D', 'F', 'D')
    
    # Impute using estimated shooter role
    df_bl = impute.impute_blocked_shot_origins(df_bl, method='cdf_mapping', role_col='est_shooter_role')
    
    # Row 3: All PBP Unblocked
    print("Preparing Unblocked Baseline (PBP)...")
    df_unb = df_pbp[df_pbp['event'].isin(['shot','goal','missed-shot'])].copy()
    df_unb = fit_xgs.enrich_data_with_bios(df_unb)
    
    # 3. Plotting
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    for col, role in enumerate(['F', 'D']):
        # --- ROW 1: All PBP Imputed (N=large) ---
        ax = axes[0, col]
        draw_rink(ax)
        # Use our proxy to filter
        subset = df_bl[df_bl['est_shooter_role'] == role]
        if not subset.empty:
            ix = np.where(subset['imputed_x'] < 0, -subset['imputed_x'], subset['imputed_x'])
            iy = np.where(subset['imputed_x'] < 0, -subset['imputed_y'], subset['imputed_y'])
            sns.kdeplot(x=ix, y=iy, cmap='Blues', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Imputed (All PBP, N={len(subset)})")
        
        # --- ROW 2: Empirical Truth (N=1060, High Score) ---
        ax = axes[1, col]
        draw_rink(ax)
        subset_emp = df_emp[df_emp['shooter_role'] == role]
        if not subset_emp.empty:
            tx = np.where(subset_emp['x'] < 0, -subset_emp['x'], subset_emp['x'])
            ty = np.where(subset_emp['x'] < 0, -subset_emp['y'], subset_emp['y'])
            sns.kdeplot(x=tx, y=ty, cmap='Greens', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Empirical (Score > 0.4, N={len(subset_emp)})")
        
        # --- ROW 3: All Unblocked (PBP) ---
        ax = axes[2, col]
        draw_rink(ax)
        subset_unb = df_unb[df_unb['shooter_role'] == role]
        if len(subset_unb) > 20000: subset_unb = subset_unb.sample(20000)
        ux = np.where(subset_unb['x'] < 0, -subset_unb['x'], subset_unb['x'])
        uy = np.where(subset_unb['x'] < 0, -subset_unb['y'], subset_unb['y'])
        sns.kdeplot(x=ux, y=uy, cmap='Reds', fill=True, alpha=0.5, ax=ax, bw_adjust=0.8)
        ax.set_title(f"{role} - Unblocked (All PBP, N={len(subset_unb)})")

    plt.tight_layout()
    plt.savefig('analysis/nested_xgs/shot_heatmaps.png')
    print("Saved to analysis/nested_xgs/shot_heatmaps.png")

if __name__ == "__main__":
    main()
