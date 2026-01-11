
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
    
    # 4. Filter for Blocked Shots
    mask_blocks = df_processed['event'] == 'blocked-shot'
    df_blocks = df_processed[mask_blocks].copy()
    print(f"Found {len(df_blocks)} blocked shots.")

    if df_blocks.empty:
        print("No blocked shots found.")
        return

    # Ensure Forward/Defense separation
    # shooter_role might be 'F', 'D', 'G', or 'Unknown'
    # We want F vs D.
    df_f = df_blocks[df_blocks['shooter_role'] == 'F']
    df_d = df_blocks[df_blocks['shooter_role'] == 'D']
    
    print(f"Forwards: {len(df_f)}")
    print(f"Defensemen: {len(df_d)}")

    # 5. Generate Heatmaps
    # We want 3 plots:
    # A. Imputed Origin (F)
    # B. Imputed Origin (D)
    # C. PBP Block Location (All) - Comparison
    
    # Imputed Coordinates: 'x_adj', 'y_adj' (which contain imputed values for blocks after pipeline)
    # PBP Coordinates: 'block_x', 'block_y' (preserved from raw)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Standardize Plot Ranges (Attack Zone to Net)
    # Rink X: -100 to 100. Attack usually > 25.
    extent = [0, 100, -42.5, 42.5] 
    
    # Helper to clean data for plotting (remove NaNs, infinite)
    def clean_coords(sub_df, x_col, y_col):
        temp = sub_df[[x_col, y_col]].dropna()
        # Filter strictly reasonable bounds to avoid plotting errors
        temp = temp[(temp[x_col] >= -100) & (temp[x_col] <= 100)]
        temp = temp[(temp[y_col] >= -45) & (temp[y_col] <= 45)]
        return temp

    # Plot A: F Imputed
    clean_f = clean_coords(df_f, 'x_adj', 'y_adj')
    if not clean_f.empty:
        # Standardize to Positive X if not already (Pipeline usually does, but double check)
        # Pipeline Step 2 standardizes.
        sns.kdeplot(
            data=clean_f, x='x_adj', y='y_adj', fill=True, cmap='Reds', ax=axes[0], levels=15, thresh=0.05
        )
        axes[0].set_title(f"Imputed Origins: Forwards (n={len(clean_f)})")
    else:
        axes[0].text(0.5, 0.5, "No Data", ha='center')

    # Plot B: D Imputed
    clean_d = clean_coords(df_d, 'x_adj', 'y_adj')
    if not clean_d.empty:
        sns.kdeplot(
            data=clean_d, x='x_adj', y='y_adj', fill=True, cmap='Blues', ax=axes[1], levels=15, thresh=0.05
        )
        axes[1].set_title(f"Imputed Origins: Defensemen (n={len(clean_d)})")
    else:
        axes[1].text(0.5, 0.5, "No Data", ha='center')

    # Plot C: PBP Locations (All)
    # Use block_x, block_y if available, else fallback to raw 'x'/'y' (which might be swapped/imputed? 
    # 'x' in df_processed IS swapped/imputed. We needed the RAW.
    # Fortunately, 'block_x' shouldn't be touched by the pipeline's x adjustment logic 
    # EXCEPT ensuring it stays with the row. It is NOT standardized orientation-wise by default 
    # unless we added that logic.
    # Wait. 'block_x' comes from 'impute.py' which takes 'x_col' (standardized x passed in?? No.)
    # In data_pipeline.py: 
    #   Step 2: Orientation Standardization (modifies 'x', 'y' in place!).
    #   Step 5: Imputation (calls impute...).
    # Inside impute.py: "df_out['block_x'] = df_out[x_col]"
    # So 'block_x' will be the STANDARDIZED x at the time of imputation.
    # This is GOOD. It means it's oriented correctly (Attack Right).
    
    clean_blocks = clean_coords(df_blocks, 'block_x', 'block_y')
    if not clean_blocks.empty:
        sns.kdeplot(
            data=clean_blocks, x='block_x', y='block_y', fill=True, cmap='Greys', ax=axes[2], levels=15, thresh=0.05
        )
        axes[2].set_title(f"PBP Block Locations (All n={len(clean_blocks)})")
    else:
        axes[2].text(0.5, 0.5, "No Data", ha='center')

    # Rink details (Simple lines)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        # Goal Line
        ax.axvline(89, color='black', linestyle='-', alpha=0.3)
        # Blue Line
        ax.axvline(25, color='blue', linestyle='-', alpha=0.3)
        # Faceoff dots (approx)
        ax.scatter([69, 69], [22, -22], color='red', s=10, alpha=0.3)

    plt.tight_layout()
    out_file = Path("analysis/blocked_shot_heatmaps_20242025.png")
    out_file.parent.mkdir(exist_ok=True)
    plt.savefig(out_file, dpi=150)
    print(f"Saved heatmaps to {out_file}")

if __name__ == "__main__":
    main()
