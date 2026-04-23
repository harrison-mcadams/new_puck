import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline, config

def main():
    print("Loading 2023-2024 season data...")
    csv_path = analyze.locate_season_csv('20232024')
    df = pd.read_csv(csv_path)
    
    # Process with the SAME pipeline used for training
    df_p = data_pipeline.preprocess_features(df, apply_filtering=True, apply_imputation=True)

    # 1. Check blocked shots
    blocks = df_p[df_p['event'] == 'blocked-shot']
    others = df_p[df_p['event'] != 'blocked-shot']

    print(f"\nTotal Blocks: {len(blocks)}")
    print(f"Total Others: {len(others)}")

    # 2. Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # X-coords
    axes[0].hist(blocks['x'], bins=50, alpha=0.5, label='Blocks', density=True)
    axes[0].hist(others['x'], bins=50, alpha=0.5, label='Others', density=True)
    axes[0].axvline(89, color='r', linestyle='--', label='Net')
    axes[0].set_title("X Coordinate Distribution (Standardized)")
    axes[0].legend()

    # Distance
    axes[1].hist(blocks['distance'], bins=50, alpha=0.5, label='Blocks', density=True)
    axes[1].hist(others['distance'], bins=50, alpha=0.5, label='Others', density=True)
    axes[1].set_title("Distance Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('analysis/block_spatial_diagnosis.png')
    print("\nSaved plot to analysis/block_spatial_diagnosis.png")

    # 3. Print stats
    print("\nBlocks X Stats:")
    print(blocks['x'].describe())
    
    print("\nOthers X Stats:")
    print(others['x'].describe())

    print("\nBlocked Probability by X Bin:")
    df_p['x_bin'] = pd.cut(df_p['x'], bins=np.linspace(0, 100, 11))
    prob = df_p.groupby('x_bin', observed=False)['event'].apply(lambda x: (x == 'blocked-shot').mean())
    print(prob)

if __name__ == "__main__":
    main()
