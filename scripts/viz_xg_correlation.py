
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def main():
    # 1. Load Data
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter: Non-Empty Net (already filtered in csv usually, but double check)
    # The previous script filtered matched_clean but maybe not saved cleanly?
    # Let's verify columns.
    
    # Filter for reasonable location matches to compare "Model vs Model" not "Bad Data vs Model"
    # User asked for "overall sense", so maybe we show ALL, but color by "Bad Data"?
    # Or just show the purely model comparison.
    # Let's show ALL matched shots (where we found a partner), but maybe handle outliers.
    
    # Let's filter slightly for visual clarity (remove Empty Net if present)
    if 'is_net_empty' in df.columns:
        df = df[df['is_net_empty'] == 0]
    if 'emptyNet' in df.columns:
        df = df[df['emptyNet'] == 0]
        
    print(f"Plotting {len(df)} non-empty net shots...")
    
    # 2. Plot
    sns.set_theme(style="whitegrid")
    
    # Correlation
    corr = df['xgs'].corr(df['xGoal'])
    mae = (df['xgs'] - df['xGoal']).abs().mean()
    
    # Use JointPlot for "Overall Sense" (Scatter + Histograms)
    g = sns.jointplot(
        data=df,
        x='xGoal', 
        y='xgs',
        kind='hex', # Hexbin is better for 50k points than scatter
        gridsize=40,
        height=8,
        marginal_kws=dict(bins=40, fill=True),
        cmap='Blues'
    )
    
    # Add 45-degree line
    g.ax_joint.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect Agreement')
    
    # Annotation
    stats_text = f"Correlation: {corr:.4f}\nMAE: {mae:.4f}\nN={len(df)}"
    g.ax_joint.text(0.05, 0.95, stats_text, transform=g.ax_joint.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    g.ax_joint.set_xlabel('MoneyPuck xG')
    g.ax_joint.set_ylabel('My Model xG')
    g.fig.suptitle(f"xG Model Comparison (2025 Season)", y=1.02, fontsize=16)
    
    # Save
    out_path = os.path.join(config.ANALYSIS_DIR, 'xg_correlation_jointplot.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved plot to {out_path}")
    
    # Also save a simple scatter for easier viewing if hex is too abstract?
    # No, hex is best for density.
    
if __name__ == "__main__":
    main()
