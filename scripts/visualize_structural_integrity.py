
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    csv_path = Path('analysis/debug_imputation_pipeline.csv')
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run train_xgboost_model.py first.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Spatial Scatter Comparison
    plt.figure(figsize=(12, 10))
    
    # Plot Unblocked Shots (Subset)
    mask_unblocked = df['event'] != 'blocked-shot'
    unblocked = df[mask_unblocked].sample(min(5000, mask_unblocked.sum()), random_state=42)
    
    # Plot Blocked Shots (Imputed)
    mask_blocked = df['event'] == 'blocked-shot'
    blocked = df[mask_blocked].sample(min(5000, mask_blocked.sum()), random_state=42)
    
    # Use Adjusted coords for unblocked if available, else x/y
    ux = unblocked['x_adj'] if 'x_adj' in unblocked.columns else unblocked['x']
    uy = unblocked['y_adj'] if 'y_adj' in unblocked.columns else unblocked['y']
    
    bx = blocked['imputed_x']
    by = blocked['imputed_y']
    
    plt.scatter(ux, uy, alpha=0.1, c='blue', s=5, label='Unblocked (Adj Origin)')
    plt.scatter(bx, by, alpha=0.1, c='red', s=5, label='Blocked (Imputed Origin)')
    
    # Draw Goal
    plt.plot(89, 0, 'ko', markersize=10, label='Goal')
    # Draw Rink Bounds roughly
    plt.axvline(25, color='k', linestyle='--', alpha=0.3, label='Blue Line')
    plt.axvline(89, color='r', linestyle='-', alpha=0.3, label='Goal Line')
    
    plt.title("Spatial Structure Comparison: Unblocked vs Imputed Blocks")
    plt.xlabel("X (ft)")
    plt.ylabel("Y (ft)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = Path('analysis/structural_integrity_scatter.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved scatter plot to {out_path}")
    
    # 2. Kernel Density Estimate (Structure Check)
    plt.figure(figsize=(12, 8))
    sns.kdeplot(x=ux, y=uy, cmap="Blues", fill=True, alpha=0.5, label='Unblocked Density')
    sns.kdeplot(x=bx, y=by, cmap="Reds", fill=False, linewidths=1.5, label='Blocked Density')
    
    plt.title("Spatial Density Comparison")
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(-42.5, 42.5)
    
    out_path_kde = Path('analysis/structural_integrity_kde.png')
    plt.savefig(out_path_kde, dpi=150)
    print(f"Saved KDE plot to {out_path_kde}")

if __name__ == "__main__":
    main()
