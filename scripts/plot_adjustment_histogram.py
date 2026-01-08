
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_histograms():
    # Use 2018-2019 data as it includes the large Tampa adjustments
    csv_path = os.path.join("data", "20182019", "20182019_df.csv")
    if not os.path.exists(csv_path):
        # Fallback to 20152016 if 2018 isn't ready
        csv_path = os.path.join("data", "20152016", "20152016_df.csv")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'x_adj' not in df.columns:
        print("Error: x_adj column not found.")
        return

    # Calculate differences as requested: (x - x_adj) and (y - y_adj)
    # This represents the "Bias" that was removed.
    df['diff_x'] = df['x'] - df['x_adj']
    df['diff_y'] = df['y'] - df['y_adj']
    
    # Filter for non-zero adjustments to see the distribution of corrections
    mask = (df['diff_x'].abs() > 0.001) | (df['diff_y'].abs() > 0.001)
    adjusted_df = df[mask]
    
    print(f"Plotting {len(adjusted_df)} adjusted events out of {len(df)} total.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # X Histogram
    axes[0].hist(adjusted_df['diff_x'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('X Correction Distribution (x - x_adj)')
    axes[0].set_xlabel('Adjustment Magnitude (ft)')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Y Histogram
    axes[1].hist(adjusted_df['diff_y'], bins=50, color='salmon', edgecolor='black')
    axes[1].set_title('Y Correction Distribution (y - y_adj)')
    axes[1].set_xlabel('Adjustment Magnitude (ft)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "adjustment_histograms.png" # Updated to png for artifacts
    plt.savefig(output_path)
    print(f"Saved histogram to {output_path}")

if __name__ == "__main__":
    plot_histograms()
