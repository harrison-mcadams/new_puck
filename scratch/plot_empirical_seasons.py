import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set modern style aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'DejaVu Sans', 'Arial']

def main():
    csv_path = Path('analysis/empirical_seasons_summary.csv')
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Format season names for better display (e.g. 20102011 -> "10-11")
    df['season_label'] = df['season'].astype(str).apply(lambda s: f"'{s[2:4]}-'{s[6:8]}")
    
    # Create a cohesive, premium color palette
    primary_color = '#3498db'  # Accuracy (Blue)
    secondary_color = '#e74c3c'  # Shooting Pct (Red)
    accent_color = '#2ecc71'  # Block Rate (Green)
    dist_color = '#9b59b6'  # Distance (Purple)
    dark_bg = '#1e272e'
    text_color = '#2c3e50'
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='#f8f9fa')
    
    # 1. Plot 1: The Great Accuracy Cliff & Block Rates
    ax1 = axes[0, 0]
    ax1.plot(df['season_label'], df['accuracy_rate'] * 100, marker='o', linewidth=3, color=primary_color, label='Accuracy Rate (On Net / Unblocked)')
    ax1.plot(df['season_label'], df['block_rate'] * 100, marker='s', linewidth=2.5, color=accent_color, linestyle='--', label='Block Rate (Blocked / Attempts)')
    ax1.set_title("Tracking Structural Shifts: The Accuracy Cliff", fontsize=14, fontweight='bold', color=text_color)
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_xlabel("Season", fontsize=12)
    ax1.set_ylim(55, 80)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    # Highlight the cliff
    ax1.axvspan("'22-'23", "'25-'26", color='#f1c40f', alpha=0.1, label='Tracking Transition')
    
    # 2. Plot 2: Shooting Percentage Surge
    ax2 = axes[0, 1]
    ax2.plot(df['season_label'], df['shooting_pct'] * 100, marker='D', linewidth=3, color=secondary_color, label='Shooting Pct (Goals / On Net)')
    ax2.set_title("Scoring Surge: Shooting Percentage on Net", fontsize=14, fontweight='bold', color=text_color)
    ax2.set_ylabel("Shooting Pct (%)", fontsize=12)
    ax2.set_xlabel("Season", fontsize=12)
    ax2.set_ylim(8.0, 11.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    # Trendline
    z = np.polyfit(range(len(df)), df['shooting_pct'] * 100, 1)
    p = np.poly1d(z)
    ax2.plot(df['season_label'], p(range(len(df))), color='#7f8c8d', linestyle=':', alpha=0.8, label='Linear Trend')
    ax2.legend(loc='upper left')

    # 3. Plot 3: Offensive Compression (Mean Distance)
    ax3 = axes[1, 0]
    ax3.plot(df['season_label'], df['mean_distance'], marker='^', linewidth=3, color=dist_color, label='Mean Shot Distance')
    ax3.set_title("Offensive Compression: Average Shot Distance", fontsize=14, fontweight='bold', color=text_color)
    ax3.set_ylabel("Distance to Goal (ft)", fontsize=12)
    ax3.set_xlabel("Season", fontsize=12)
    ax3.set_ylim(31.5, 35.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # 4. Plot 4: Attempt Volume Scale
    ax4 = axes[1, 1]
    ax4.bar(df['season_label'], df['total_attempts'] / 1000, color='#34495e', alpha=0.8, width=0.6, label='Shot Attempts (Thousands)')
    ax4.set_title("Dataset Scale: Total Shot Attempts per Season", fontsize=14, fontweight='bold', color=text_color)
    ax4.set_ylabel("Attempts (Thousands)", fontsize=12)
    ax4.set_xlabel("Season", fontsize=12)
    ax4.set_ylim(50, 180)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Add a main aesthetic title
    plt.suptitle("NHL Empirical Shot Tracking & Scoring Trends (2010–2026)\nAn analysis of tracking anomalies, offensive compression, and modern era surge", 
                 fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the dashboard
    out_dir = Path('analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'empirical_seasons_trends.png'
    plt.savefig(out_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    
    print(f"Success! Trends dashboard successfully plotted and saved to: {out_path}")

if __name__ == "__main__":
    main()
