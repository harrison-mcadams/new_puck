
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import rink
from puck import config

def plot_static_vectors():
    if len(sys.argv) > 2:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
    else:
        target_game_id = 2024020202
        target_goal_id = 328
    
    # Correct Output Dir
    base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "new_puck")
    out_dir = os.path.join(base_dir, 'analysis', 'blocked_shots')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data
    # CSV was saved in data/analysis/candidate_shot_vectors.csv
    # But wait, previous script said: os.path.join(base_dir, 'analysis', ...) where base_dir=config.DATA_DIR
    csv_path = os.path.join(config.DATA_DIR, 'analysis', 'candidate_shot_vectors.csv')
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df_cand = pd.read_csv(csv_path)
    
    # Normalize if needed?
    # The analyze script saved what it had.
    # Analyze script loaded `df_puck` from `data/...`.
    # AND it calculated `dist_shooter`.
    # It probably has RAW coordinates because I didn't see explicit normalization in `analyze_shot_vector_candidates.py`.
    # Let's check ranges.
    
    if df_cand['x'].abs().max() > 200:
        print("Normalizing Coordinates...")
        df_cand['x'] = (df_cand['x'] - 1200.0) / 12.0
        df_cand['y'] = -(df_cand['y'] - 510.0) / 12.0
        df_cand['vx'] = df_cand['vx'] / 12.0
        df_cand['vy'] = df_cand['vy'] / -12.0

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rink.draw_rink(ax)
    
    # Plot Vectors
    # Color by Dev? Or Time?
    # Let's use Time (Frame Index) to show progression
    
    scatter = ax.scatter(df_cand['x'], df_cand['y'], c=df_cand['frame_idx'], cmap='viridis', zorder=5)
    plt.colorbar(scatter, label='Frame Index')
    
    for _, row in df_cand.iterrows():
        # Draw Arrow
        dx = row['vx'] * 0.5 # Scale for visibility
        dy = row['vy'] * 0.5
        
        # Color based on deviation?
        # Red if good deviation (<10), Grey otherwise?
        color = 'red' if row['dev_deg'] < 10 else 'gray'
        alpha = 1.0 if row['dev_deg'] < 10 else 0.5
        width = 0.02 if row['dev_deg'] < 10 else 0.01
        
        ax.arrow(row['x'], row['y'], dx, dy, 
                 head_width=1, head_length=1.5, fc=color, ec=color, width=0.3, alpha=alpha, zorder=4)
        
        if row['dev_deg'] < 10:
             ax.text(row['x'], row['y']+1, f"F{int(row['frame_idx'])}", color='black', fontsize=8, fontweight='bold')

    ax.set_title(f"Candidate Shot Vectors | Game {target_game_id} Goal {target_goal_id}")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    
    out_file = os.path.join(out_dir, f'all_vectors_{target_game_id}_{target_goal_id}.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Static plot saved to {out_file}")

if __name__ == "__main__":
    plot_static_vectors()
