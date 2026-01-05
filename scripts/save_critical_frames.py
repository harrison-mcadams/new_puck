
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from puck import rink, config

def save_frames():
    target_game_id = 2024020118
    target_goal_id = 319
    frames_of_interest = [8, 26, 43, 44, 45, 50]
    
    # Load Data
    base_dir = config.DATA_DIR
    data_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{target_game_id}_goal_{target_goal_id}_positions.csv')
    df = pd.read_csv(data_path)
    
    # Normalize
    def norm_x(x): return (x - 1200.0) / 12.0
    def norm_y(y): return -(y - 510.0) / 12.0
    
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    if col_id in df.columns: df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
    
    if df['x'].max() > 1000:
        df['x'] = norm_x(df['x'])
        df['y'] = norm_y(df['y'])
        
    df_puck = df[df['entity_type'] == 'puck']
    df_players = df[df['entity_type'] == 'player']
    
    out_dir = os.path.join(os.path.expanduser("~"), ".gemini/antigravity/brain/3be1d619-2259-42ba-9092-b0163917e8ec")
    
    for f in frames_of_interest:
        fig, ax = plt.subplots(figsize=(12, 6))
        rink.draw_rink(ax)
        
        # Players
        fp = df_players[df_players['frame_idx'] == f]
        ax.scatter(fp['x'], fp['y'], c='blue', s=50, alpha=0.6)
        
        # Puck
        fk = df_puck[df_puck['frame_idx'] == f]
        if not fk.empty:
            ax.scatter(fk['x'], fk['y'], c='black', s=80, edgecolors='white', zorder=5)
            # Label
            ax.text(fk['x'].iloc[0], fk['y'].iloc[0]+2, f"Puck", color='black', fontsize=8)
            
        ax.set_title(f"Game {target_game_id} | Frame {f}")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        
        out_path = os.path.join(out_dir, f"frame_{f}.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    save_frames()
