
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
import requests

def get_goal_metadata(game_id, goal_id):
    """
    Fetches scorer name, time and period for a given goal event.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200: return None
        data = resp.json()
    except Exception as e:
        print(f"  [API Error] {e}")
        return None

    plays = data.get('plays', [])
    goal_play = next((p for p in plays if str(p.get('eventId')) == str(goal_id)), None)
    
    if not goal_play: return None
    
    details = goal_play.get('details', {})
    scorer_id = details.get('scoringPlayerId')
    scorer_name = "Unknown"
    
    if scorer_id:
        try:
            p_resp = requests.get(f"https://api-web.nhle.com/v1/player/{scorer_id}/landing", timeout=5).json()
            fname = p_resp.get('firstName', {}).get('default', '')
            lname = p_resp.get('lastName', {}).get('default', '')
            scorer_name = f"{fname} {lname}".strip()
        except:
            pass
            
    return {
        'scorer': scorer_name,
        'time': goal_play.get('timeInPeriod'),
        'period': goal_play.get('periodDescriptor', {}).get('number')
    }

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
    candidates_path = os.path.join(config.ANALYSIS_DIR, 'blocked_shots', f'candidate_vectors_{target_game_id}_{target_goal_id}_v2.csv')
    
    if not os.path.exists(data_path) or not os.path.exists(candidates_path):
        print("Data files not found.")
        return

    df = pd.read_csv(data_path)
    df_cand = pd.read_csv(candidates_path)
    
    meta = get_goal_metadata(target_game_id, target_goal_id)
    meta_str = ""
    if meta:
        meta_str = f"{meta['scorer']} | {meta['time']} P{meta['period']}"
    
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    
    # Frames
    min_f = int(df['frame_idx'].min())
    max_f = int(min(df['frame_idx'].max(), 100)) # Limit to first 100 frames
    
    print(f"Generating animation frames {min_f} to {max_f}...")

    def update(frame_idx):
        ax.clear()
        rink.draw_rink(ax)
        
        # Plot Players
        frame_players = df_players[df_players['frame_idx'] == frame_idx]
        
        # Check for Blocker ID in candidates (take first row, assuming same block ID)
        blocker_id = None
        if 'blocker_id' in df_cand.columns and not df_cand.empty:
             val = df_cand.iloc[0]['blocker_id']
             if pd.notnull(val): blocker_id = val
             
        # Plot Regular Players
        if blocker_id:
             others = frame_players[frame_players[col_id] != blocker_id]
             blocker = frame_players[frame_players[col_id] == blocker_id]
             ax.scatter(others['x'], others['y'], c='blue', s=50, alpha=0.6, label='Players')
             ax.scatter(blocker['x'], blocker['y'], c='magenta', marker='s', s=80, edgecolors='black', label='Blocker')
        else:
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
            
            ax.text(row['x'], row['y']+2, f"F{int(row['frame_idx'])}\nDev:{row['dev_deg']:.1f}\nBlk:{row['blocker_dist']:.1f}ft", 
                    color='red', fontsize=8, fontweight='bold')
            
        title = f"Game {target_game_id} | Goal {target_goal_id}"
        if meta_str:
            title += f"\n{meta_str}"
        title += f" | Frame {frame_idx}"
        ax.set_title(title)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)

    ani = animation.FuncAnimation(fig, update, frames=range(min_f, max_f), interval=100)
    
    # Save to analysis/blocked_shots
    out_file = os.path.join(base_dir, '..', 'analysis', 'blocked_shots', f'candidate_vectors_{target_game_id}_{target_goal_id}.gif')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    ani.save(out_file, writer='pillow', fps=10)
    print(f"Animation saved to {out_file}")

    # Save Best Candidate PNG
    if not df_cand.empty and 'score' in df_cand.columns:
        best_row = df_cand.loc[df_cand['score'].idxmax()]
        # If best_row is a DataFrame (multiple maxes), take first
        if isinstance(best_row, pd.DataFrame):
            best_row = best_row.iloc[0]
            
        best_frame = int(best_row['frame_idx'])
        print(f"Saving Best Candidate PNG for Frame {best_frame} (Score {best_row['score']:.3f})...")
        
        # Update plot to that frame
        update(best_frame)
        png_out = os.path.join(base_dir, '..', 'analysis', 'blocked_shots', f'candidate_vectors_{target_game_id}_{target_goal_id}_best.png')
        plt.savefig(png_out)
        print(f"PNG saved to {png_out}")

if __name__ == "__main__":
    visualize_candidates()
