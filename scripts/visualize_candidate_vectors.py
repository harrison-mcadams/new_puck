
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import rink
from puck import config

def visualize_candidates():
    # Config
    if len(sys.argv) > 2:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
    else:
        target_game_id = 2024020202
        target_goal_id = 328
    
    # Load Data
    base_dir = config.DATA_DIR
    data_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{target_game_id}_goal_{target_goal_id}_positions.csv')
    candidates_path = os.path.join(base_dir, 'analysis', 'candidate_shot_vectors.csv')
    
    if not os.path.exists(data_path) or not os.path.exists(candidates_path):
        print("Data files not found.")
        return

    df = pd.read_csv(data_path)
    df_cand = pd.read_csv(candidates_path)
    
    # Normalize coords if needed (assume standard -100 to 100 range required by rink.draw_rink?)
    # The data seems to be raw 0-200ft or similar. 
    # Let's check max values.
    # If x > 200, it's raw.
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    if col_id in df.columns: df[col_id] = pd.to_numeric(df[col_id], errors='coerce')

    # Puck Data
    if 'entity_type' in df.columns:
        df_puck = df[df['entity_type'] == 'puck'].copy().sort_values('frame_idx')
        df_players = df[df['entity_type'] == 'player'].copy()
    else:
        df_puck = df[df[col_id].isnull()].copy().sort_values('frame_idx')
        df_players = df[df[col_id].notnull()].copy()

    # Normalize
    if df_puck['x'].abs().max() > 200:
         df['x'] = (df['x'] - 1200.0) / 12.0
         df['y'] = -(df['y'] - 510.0) / 12.0
         # Re-split
         df_puck = df[df['entity_type'] == 'puck']
         df_players = df[df['entity_type'] == 'player']
         # Note: candidates CSV has x,y. They might be in normalized or raw?
         # scripts/analyze_shot_vector_candidates.py operated on normalized df_puck if logical...
         # Wait, analyze_vectors didn't normalize explicitly in the script I wrote.
         # It checked `df_puck['x'].diff()` etc.
         # So candidates CSV probably has Raw coords if I didn't normalize in analyze script.
         # Actually, analyze script calculated speed/angle. 
         # Let's normalize candidate coords here just in case they are raw.
         # Check a sample value
         pass 

    # We need to ensure consistency. 
    # Let's assume input CSVs are consistent. 
    # If the candidate script ran on Raw data, then we need to Normalize both here.
    
    # Check Analyze Script again:
    # It loaded data. It did NOT include the normalization block I usually put (lines 30-36 in trace script).
    # So analyze_vectors used RAW coordinates (0-2400 approx).
    # This means Speed was in RAW units / 0.1s.
    
    # Rink plotting functions usually expect Feet (-100 to 100).
    # So let's Normalize Everything.
    
    # Function to normalize raw
    def norm_x(x): return (x - 1200.0) / 12.0
    def norm_y(y): return -(y - 510.0) / 12.0
    
    if df['x'].max() > 1000: # It's raw
        df['x'] = norm_x(df['x'])
        df['y'] = norm_y(df['y'])
        
        df_cand['x'] = norm_x(df_cand['x'])
        df_cand['y'] = norm_y(df_cand['y'])
        # VX/VY in candidates need scaling too?
        # vx = dx/0.1. dx is scaled by 1/12. So vx is scaled by 1/12.
        df_cand['vx'] = df_cand['vx'] / 12.0
        df_cand['vy'] = df_cand['vy'] / -12.0 # Invert Y?
        
    df_puck = df[df['entity_type'] == 'puck']
    df_players = df[df['entity_type'] == 'player']

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Frames
    min_f = int(df['frame_idx'].min())
    max_f = int(min(df['frame_idx'].max(), 100)) # Limit to first 100 frames
    
    print(f"Generating animation frames {min_f} to {max_f}...")

    def update(frame_idx):
        ax.clear()
        rink.draw_rink(ax)
        
        # Plot Players
        frame_players = df_players[df_players['frame_idx'] == frame_idx]
        ax.scatter(frame_players['x'], frame_players['y'], c='blue', s=50, alpha=0.6, label='Players')
        
        # Plot Puck
        frame_puck = df_puck[df_puck['frame_idx'] == frame_idx]
        if not frame_puck.empty:
            ax.scatter(frame_puck['x'], frame_puck['y'], c='black', s=80, edgecolors='white', zorder=5, label='Puck')
            
        # Plot Candidate Vectors (if any for this frame)
        # We also want to see Past candidates fading or specific ones?
        # User said "in the setting of real time".
        # Let's show the vector *originating* at this frame.
        
        # Check if this frame is a candidate start
        matches = df_cand[df_cand['frame_idx'] == frame_idx]
        for _, row in matches.iterrows():
            # Draw Vector
            # Scale length
            speed_scale = 0.5
            dx = row['vx'] * speed_scale
            dy = row['vy'] * speed_scale
            
            ax.arrow(row['x'], row['y'], dx, dy, 
                     head_width=2, head_length=3, fc='red', ec='red', width=0.5, zorder=10)
            
            ax.text(row['x'], row['y']+2, f"F{int(row['frame_idx'])}\nDev:{row['dev_deg']:.1f}", 
                    color='red', fontsize=8, fontweight='bold')
            
        ax.set_title(f"Game {target_game_id} | Frame {frame_idx}")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)

    ani = animation.FuncAnimation(fig, update, frames=range(min_f, max_f), interval=100)
    
    # Save to analysis/blocked_shots
    out_file = os.path.join(base_dir, '..', 'analysis', 'blocked_shots', f'candidate_vectors_{target_game_id}_{target_goal_id}.gif')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    ani.save(out_file, writer='pillow', fps=10)
    print(f"Animation saved to {out_file}")

if __name__ == "__main__":
    visualize_candidates()
