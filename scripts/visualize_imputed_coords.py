
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root
sys.path.append(os.getcwd())
from puck import fit_xgs, impute

def main():
    print("Loading data...")
    # Load 2024 data directly to avoid signature issues
    df = pd.read_csv('data/20242025/20242025_df.csv')
    
    print("Imputing blocked shots...")
    # Force CDF method and ensure we are using unadjusted x,y if adjusted not avail, or whatever the default is
    # We want to test the imputation logic itself.
    
    # Filter to blocked only
    df_blocked = df[df['event'] == 'blocked-shot'].copy()
    
    # Impute
    df_blocked = impute.impute_blocked_shot_origins(df_blocked, method='cdf_mapping')
    
    # Filter to unblocked shots (Wrist/Msg/Gol)
    df_unblocked = df[df['event'].isin(['shot', 'goal', 'missed-shot'])].copy()
    
    print(f"Blocked (Imputed): {len(df_blocked)}")
    print(f"Unblocked (Actual): {len(df_unblocked)}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    
    # 1. Unblocked Shots (Target Distribution)
    axes[0].set_title("Actual Unblocked Shot Locations")
    sns.scatterplot(data=df_unblocked, x='x', y='y', s=2, alpha=0.1, ax=axes[0], color='blue')
    # Add rink density contour
    sns.kdeplot(data=df_unblocked, x='x', y='y', levels=5, color='white', linewidths=0.5, ax=axes[0], alpha=0.5)

    # 2. Imputed Blocked Shots (Generated Distribution)
    axes[1].set_title("Imputed Blocked Shot Origins (CDF Method)")
    sns.scatterplot(data=df_blocked, x='imputed_x', y='imputed_y', s=2, alpha=0.1, ax=axes[1], color='red')
    sns.kdeplot(data=df_blocked, x='imputed_x', y='imputed_y', levels=5, color='white', linewidths=0.5, ax=axes[1], alpha=0.5)

    # Rink bounds for context
    import matplotlib.patches as patches
    for ax in axes:
        ax.set_xlim(25, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        # Goal
        ax.add_patch(patches.Circle((89, 0), 1, color='green'))

    plt.tight_layout()
    out_path = 'analysis/imputation_verification.png'
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
