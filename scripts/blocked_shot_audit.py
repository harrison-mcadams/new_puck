
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
try:
    from puck import rink
    from puck import impute
except ImportError:
    pass

def corrected_impute_row(row, d=5.64):
    bx, by = row['x'], row['y']
    dist = row['distance']
    
    # We need to find which goal was attacked.
    # parse.py uses (89, 0) or (-89, 0).
    # d1 = sqrt((bx-89)**2 + by**2)
    # d2 = sqrt((bx+89)**2 + by**2)
    
    d1 = np.sqrt((bx - 89)**2 + by**2)
    d2 = np.sqrt((bx + 89)**2 + by**2)
    
    if abs(d1 - dist) < abs(d2 - dist):
        goal_x = 89
    else:
        goal_x = -89
        
    # Vector from Net to Block
    vx = bx - goal_x
    vy = by - 0 # goal_y
    
    mag = np.sqrt(vx**2 + vy**2)
    if mag == 0:
        return bx, by, dist
        
    ux = vx / mag
    uy = vy / mag
    
    ox = bx + (ux * d)
    oy = by + (uy * d)
    
    new_dist = np.sqrt((ox - goal_x)**2 + oy**2)
    
    return ox, oy, new_dist

def main():
    data_file = Path('data/20252026.csv')
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    
    # Filter for all shots
    all_shots = df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])].copy()
    
    blocks = all_shots[all_shots['event'] == 'blocked-shot'].copy()
    others = all_shots[all_shots['event'] != 'blocked-shot'].copy()
    
    print(f"Total shots: {len(all_shots)}")
    print(f"Blocks: {len(blocks)}")
    
    # Apply CURRENT (BORKED) imputation to blocks
    borked_blocks = impute.impute_blocked_shot_origins(blocks, method='mean_6')
    
    # Apply CORRECTED imputation to blocks
    corrected_results = blocks.apply(lambda r: corrected_impute_row(r), axis=1)
    blocks['imputed_x_corr'] = [x[0] for x in corrected_results]
    blocks['imputed_y_corr'] = [x[1] for x in corrected_results]
    blocks['distance_corr'] = [x[2] for x in corrected_results]

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Nominal locations of all shots
    ax = axes[0, 0]
    rink.draw_rink(ax)
    ax.scatter(others['x'], others['y'], s=5, alpha=0.1, color='blue', label='SOG/Goal/Miss')
    ax.scatter(blocks['x'], blocks['y'], s=5, alpha=0.3, color='red', label='Block (Nominal)')
    ax.set_title("Nominal Locations")
    ax.legend()
    
    # Plot 2: Distances
    ax = axes[0, 1]
    ax.hist(others['distance'], bins=50, alpha=0.5, label='SOG/Goal/Miss', density=True)
    ax.hist(blocks['distance'], bins=50, alpha=0.5, label='Block (Nominal)', density=True)
    ax.set_title("Distance Distribution (Nominal)")
    ax.set_xlabel("Distance from Net")
    ax.legend()
    
    # Plot 3: Borked vs Corrected Imputation (X-coordinates)
    ax = axes[1, 0]
    ax.hist(blocks['distance'], bins=50, alpha=0.3, label='Nominal', density=True)
    ax.hist(borked_blocks['distance'], bins=50, alpha=0.3, label='Borked Impute', density=True)
    ax.hist(blocks['distance_corr'], bins=50, alpha=0.3, label='Corrected Impute', density=True)
    ax.set_title("Distance Distribution: Nominal vs Imputed")
    ax.set_xlabel("Distance")
    ax.legend()
    
    # Plot 4: Scatter of change
    ax = axes[1, 1]
    rink.draw_rink(ax)
    # Subset for clearer arrows
    sample = blocks.sample(min(100, len(blocks)))
    for i, row in sample.iterrows():
        ax.arrow(row['x'], row['y'], row['imputed_x_corr'] - row['x'], row['imputed_y_corr'] - row['y'], 
                 head_width=1, head_length=1, fc='green', ec='green', alpha=0.5)
    ax.set_title("Corrected Imputation Vectors (Sample)")

    plt.tight_layout()
    plt.savefig('analysis/blocked_shot_audit.png')
    print("Saved audit plot to analysis/blocked_shot_audit.png")
    
    # Summary Table
    print("\nDistance Summary:")
    summary = pd.DataFrame({
        'SOG/Goal/Miss': others['distance'].describe(),
        'Block (Nominal)': blocks['distance'].describe(),
        'Block (Current BORKED)': borked_blocks['distance'].describe(),
        'Block (Corrected)': blocks['distance_corr'].describe()
    })
    print(summary)
    
    # Check "Borked" effect on left-goal shots
    left_blocks = blocks[blocks['x'] < 0]
    if len(left_blocks) > 0:
        print("\nLeft-goal Blocked Shots (x < 0):")
        print(f"Mean distance (Nominal): {left_blocks['distance'].mean():.2f}")
        print(f"Mean distance (Borked): {borked_blocks.loc[left_blocks.index, 'distance'].mean():.2f}")
        print(f"Mean distance (Corrected): {left_blocks['distance_corr'].mean():.2f}")

if __name__ == "__main__":
    main()
