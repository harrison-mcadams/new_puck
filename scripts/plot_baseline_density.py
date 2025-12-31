import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.rink import draw_rink

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
OUTPUT_DIR = os.path.join("analysis", "gravity")
BASELINE_FILE = os.path.join(DATA_DIR, "mod_baseline.csv")

def plot_polished_baseline():
    if not os.path.exists(BASELINE_FILE):
        print(f"Baseline file not found: {BASELINE_FILE}")
        return

    df = pd.read_csv(BASELINE_FILE)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(15, 10))
    draw_rink(ax)
    
    # We only care about the offensive zone for this baseline
    # X from 0 to 100
    # Y from -42.5 to 42.5
    
    # Custom colormap: Red (High Pressure/Low MOD) to Green (Low Pressure/High MOD)
    colors = ["#ff4b2b", "#ffb74d", "#81c784"] # Red -> Orange -> Green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("gravity", colors, N=n_bins)
    
    # Pivot for heatmap
    pivot = df.pivot(index='y_bin', columns='x_bin', values='mean')
    
    # Plot heatmap
    im = ax.imshow(
        pivot, 
        extent=[df['x_bin'].min(), df['x_bin'].max() + 5, df['y_bin'].min(), df['y_bin'].max() + 5],
        origin='lower',
        cmap=cmap.reversed(), # Reverse so Red is LOW MOD (High Pressure)
        alpha=0.7,
        zorder=1
    )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Expected Mean Opponent Distance (MOD) - feet', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_title("Defensive Density Baseline: Expected MOD by Rink Location\n(Based on ~400 Goal Sequences)", fontsize=18, pad=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(-42.5, 42.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "defensive_density_baseline.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved polished baseline to {out_file}")

if __name__ == "__main__":
    plot_polished_baseline()
