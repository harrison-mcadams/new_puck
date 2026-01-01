
import pandas as pd
import numpy as np
import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck.possession import infer_possession_events

def check_player_loc(player_name_fragment):
    print(f"Searching for player: {player_name_fragment}...")
    
    # 1. Find Player ID
    # Use analysis file as index
    try:
        idx_df = pd.read_csv('analysis/gravity/player_gravity_season.csv')
        matches = idx_df[idx_df['player_name'].str.contains(player_name_fragment, case=False, na=False)]
        
        if matches.empty:
            print("Player not found in analysis.")
            return
            
        # Prioritize most recent season, most goals
        best_match = matches.sort_values('goals_on_ice_count', ascending=False).iloc[0]
        pid = str(best_match['player_id'])
        full_name = best_match['player_name']
        print(f"Analyzing {full_name} (ID: {pid})...")
        
    except Exception as e:
        print(f"Error checking index: {e}")
        return

    # 2. Scan Files
    # (Simplified scan: check a sample of games to be fast)
    files = glob.glob(r"data/edge_goals/20252026/*_positions.csv") # Check current season
    if not files:
        files = glob.glob(r"data/edge_goals/20242025/*_positions.csv")
    
    # Limit to first 20 valid files for speed
    
    valid_files = 0
    dists_to_net = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Normalize to 0-100 x, 0-85 y?
            # Standard NHL: 200x85. Goal at 89. Center at 0.
            # My coords are usually centered.
            # Goals are at X=89 and X=-89.
            # Attacking end logic needed? 
            # Yes, check if their X is mostly positive or negative.
            
            # Filter to player
            p_df = df[(df['entity_type'] == 'player') & (df['entity_id'].astype(str) == pid)]
            if p_df.empty: continue
            
            # Attacking End Normalization (Simple Heuristic for Goal Events)
            # In a goal event, the scorer is usually near the attacking net.
            # BUT we don't know who scored.
            # Heuristic: The PUCK ends up in the net.
            # Puck X should be > 80 or < -80 at end.
            puck_df = df[df['entity_type'] == 'puck']
            if not puck_df.empty:
                last_x = puck_df.iloc[-1]['x']
                attack_zoom = 1.0
                if last_x < -25: attack_zoom = -1.0 # Scoring on Left Net
                elif last_x > 25: attack_zoom = 1.0 # Scoring on Right Net
                else: continue # Mid-ice goal? Skip.
            else:
                continue

            # Transform Player to Attacking Frame (Net at X=89, Y=0)
            xs = p_df['x'] * attack_zoom
            ys = p_df['y'] * attack_zoom # Y flip matters less for distance but good for side
            
            # Off-Puck Filter
            poss_events = infer_possession_events(df, threshold_ft=6.0)
            poss_frames = set()
            for _, pev in poss_events.iterrows():
                if str(pev['player_id']) == pid:
                    for fr in range(int(pev['start_frame']), int(pev['end_frame']) + 1):
                        poss_frames.add(fr)
            
            # Off Puck Coords
            is_off = ~p_df['frame_idx'].isin(poss_frames)
            
            off_xs = xs[is_off]
            off_ys = ys[is_off]
            
            # Calculate Distance to Net (89, 0)
            # Actually net is at 89.
            dists = np.sqrt((off_xs - 89)**2 + (off_ys - 0)**2)
            
            dists_to_net.extend(dists.tolist())
            
            valid_files += 1
            if valid_files >= 30: break
            
        except: pass
        
    if not dists_to_net:
        print("No valid off-puck data found.")
        return

    avg_dist = np.mean(dists_to_net)
    print(f"Analysis Complete (n={valid_files} goals):")
    print(f"Average Off-Puck Distance to Net: {avg_dist:.1f} ft")
    
    # Interpretation
    if avg_dist < 20: print("Role: Net Front / Slot (High Danger)")
    elif avg_dist < 40: print("Role: High Slot / Circle (Scoring Danger)")
    else: print("Role: Perimeter / Point (Low Danger)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_player_loc(sys.argv[1])
    else:
        check_player_loc("Draisaitl")
