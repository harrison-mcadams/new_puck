import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    csv_path = "analysis/season_shots_20252026.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    # Use a hexbin or scatter plot with transparency because there are 105k points
    plt.hexbin(df['xg_nested'], df['xtg_mixed'], gridsize=50, cmap='inferno', mincnt=1)
    cb = plt.colorbar(label='Count')
    
    # Add unity line
    max_val = max(df['xg_nested'].max(), df['xtg_mixed'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='x=y (No Change)')
    
    plt.xlabel('Base xG (Nested Tensor)')
    plt.ylabel('Mixed Effects xtG')
    plt.title('Base xG vs Mixed Effects xtG for 2025-2026 Season Shots')
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = "analysis/xgs/xg_vs_xtg_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
