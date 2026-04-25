import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline, rink

def main():
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    dfs = []
    
    print("Loading modern era data...")
    for s in seasons:
        csv_path = analyze.locate_season_csv(s)
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Apply preprocessing to get coordinates standardized
            df_p = data_pipeline.preprocess_features(df, apply_filtering=True, verbose=False)
            dfs.append(df_p)
    
    all_df = pd.concat(dfs)
    print(f"Total rows: {len(all_df)}")
    
    # Filter for Blocked Shots
    blocks = all_df[all_df['event'] == 'blocked-shot'].copy()
    print(f"Total blocks: {len(blocks)}")
    
    # Check roles
    print("Blocks by role:")
    print(blocks['shooter_role'].value_counts())
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    roles = ['F', 'D']
    
    for i, role in enumerate(roles):
        subset = blocks[blocks['shooter_role'] == role]
        ax = axes[i]
        
        # Create 2D histogram
        h = ax.hexbin(subset['x'], subset['y'], gridsize=30, cmap='YlOrRd', mincnt=1)
        fig.colorbar(h, ax=ax)
        
        ax.set_title(f"Blocked Shot Locations - {role}")
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        
        # Add goal line
        ax.axvline(89, color='red', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plot_path = "analysis/xgboost_alternate_xgs/blocked_shot_heatmaps.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
