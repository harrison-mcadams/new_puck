
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data_file = Path('data/20252026.csv')
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows from {data_file}")

    blocks = df[df['event'] == 'blocked-shot']
    print(f"Total blocked shots: {len(blocks)}")
    
    if len(blocks) == 0:
        return

    print("\nBlocked Shot X-coordinate stats:")
    print(blocks['x'].describe())

    # Check if we have shots at both ends
    print("\nAll shots X-coordinate stats:")
    print(df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])]['x'].describe())

    plt.figure(figsize=(10, 6))
    plt.hist(df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])]['x'], bins=50, alpha=0.5, label='All Shots')
    plt.hist(blocks['x'], bins=50, alpha=0.5, label='Blocked Shots')
    plt.axvline(89, color='r', linestyle='--', label='Net X=89')
    plt.axvline(-89, color='b', linestyle='--', label='Net X=-89')
    plt.legend()
    plt.title("Distribution of X coordinates for shots")
    plt.savefig('analysis/shot_x_dist.png')
    print("\nSaved histogram to analysis/shot_x_dist.png")

    # Now let's see Distance vs X
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['distance'], alpha=0.1, s=1)
    plt.title("Distance vs X coordinate")
    plt.xlabel("X")
    plt.ylabel("Distance")
    plt.savefig('analysis/dist_vs_x.png')
    print("Saved scatter to analysis/dist_vs_x.png")

if __name__ == "__main__":
    main()
