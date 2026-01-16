
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import data_pipeline, enrich

def main():
    # 1. Load Data
    data_path = Path("data/20242025/20242025_df.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    print("Loading 2024-2025 data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # 2. Enrich (Backfill shooter_role)
    print("Enriching data (shooter_role/shoots_catches)...")
    enricher = enrich.PlayerEnricher()
    # This fetches data from API if missing, ensuring we have roles for F/D split
    df = enricher.enrich_dataframe(df, target_cols=['shoots_catches', 'shooter_role'])
    
    # 3. Process through Pipeline
    print("Running Data Pipeline...")
    # apply_imputation=True is critical
    df_processed = data_pipeline.preprocess_features(
        df,
        is_training=False,
        apply_filtering=False, # We filter manually later to ensure we have blocks
        apply_imputation=True,
        apply_arena_adjustments=True,
        verbose=True
    )
    
    # 4. Filters
    # Unblocked: SOG, Miss, Goal
    mask_unblocked = df_processed['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    mask_blocks = df_processed['event'] == 'blocked-shot'
    
    df_unblocked = df_processed[mask_unblocked].copy()
    df_blocks = df_processed[mask_blocks].copy()
    
    print(f"Unblocked Shots: {len(df_unblocked)}")
    print(f"Blocked Shots:   {len(df_blocks)}")

    if df_blocks.empty:
        print("No blocked shots found.")
        return

    # 5. Generate Comparison Heatmaps (F vs D)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    extent = [0, 100, -42.5, 42.5]
    
    def clean_coords(sub_df):
        temp = sub_df[['x_adj', 'y_adj']].dropna()
        temp = temp[(temp['x_adj'] >= -100) & (temp['x_adj'] <= 100)]
        temp = temp[(temp['y_adj'] >= -45) & (temp['y_adj'] <= 45)]
        # Filter for offensive zone roughly
        return temp

    # -- Forward Comparison --
    ax_f = axes[0]
    
    # Unblocked F
    f_unblocked = df_unblocked[df_unblocked['shooter_role'] == 'F']
    clean_f_unb = clean_coords(f_unblocked)
    if not clean_f_unb.empty:
        sns.kdeplot(
            data=clean_f_unb, x='x_adj', y='y_adj', 
            fill=False, color='blue', linewidths=2, alpha=0.8,
            ax=ax_f, levels=8, thresh=0.05, label='Unblocked (Blue)'
        )
    
    # Imputed F
    f_blocks = df_blocks[df_blocks['shooter_role'] == 'F']
    clean_f_blk = clean_coords(f_blocks)
    if not clean_f_blk.empty:
        sns.kdeplot(
            data=clean_f_blk, x='x_adj', y='y_adj', 
            fill=True, cmap='Reds', alpha=0.5,
            ax=ax_f, levels=8, thresh=0.05, label='Imputed Blocks (Red)'
        )

    ax_f.set_title(f"Forwards: Unblocked (n={len(clean_f_unb)}) vs Imputed (n={len(clean_f_blk)})")
    ax_f.legend()

    # -- Defense Comparison --
    ax_d = axes[1]
    
    # Unblocked D
    d_unblocked = df_unblocked[df_unblocked['shooter_role'] == 'D']
    clean_d_unb = clean_coords(d_unblocked)
    if not clean_d_unb.empty:
        sns.kdeplot(
            data=clean_d_unb, x='x_adj', y='y_adj', 
            fill=False, color='green', linewidths=2, alpha=0.8,
            ax=ax_d, levels=8, thresh=0.05, label='Unblocked (Green)'
        )
    
    # Imputed D
    d_blocks = df_blocks[df_blocks['shooter_role'] == 'D']
    clean_d_blk = clean_coords(d_blocks)
    if not clean_d_blk.empty:
        sns.kdeplot(
            data=clean_d_blk, x='x_adj', y='y_adj', 
            fill=True, cmap='Purples', alpha=0.5,
            ax=ax_d, levels=8, thresh=0.05, label='Imputed Blocks (Purple)'
        )

    ax_d.set_title(f"Defensemen: Unblocked (n={len(clean_d_unb)}) vs Imputed (n={len(clean_d_blk)})")
    ax_d.legend()

    # Rink details
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        ax.axvline(89, color='black', linestyle='-', alpha=0.3) # Goal Line
        ax.axvline(25, color='blue', linestyle='-', alpha=0.3)  # Blue Line
        ax.scatter([69, 69], [22, -22], color='red', s=10, alpha=0.3) # Dots

    plt.tight_layout()
    out_file = Path("analysis/blocked_shot_comparison_heatmaps.png")
    out_file.parent.mkdir(exist_ok=True)
    plt.savefig(out_file, dpi=150)
    print(f"Saved comparison heatmaps to {out_file}")

if __name__ == "__main__":
    main()
