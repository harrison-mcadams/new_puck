
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck.nhl_api import get_game_feed
from puck.edge import transform_coordinates, filter_data_to_goal_moment
from puck.possession import infer_possession_events

def check_loc():
    # Load his raw events from the metadata/analysis intersection is hard without re-processing.
    # Easier: Just process the 2 events we KNOW are his from the "grebenkin_events.csv" logic, 
    # OR simpler: re-scan the season for his player ID 8483733 and just print the coords.
    
    # Grebenkin ID: 8483733
    TARGET_PID = "8483733"
    
    import glob
    files = glob.glob(r"data/edge_goals/20252026/*_positions.csv")
    
    off_puck_x = []
    off_puck_y = []
    
    print(f"Scanning {len(files)} files for Player {TARGET_PID}...")
    
    count = 0
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normalize units if needed
            if df['x'].abs().max() > 120:
                 df['x'] = (df['x'] - 1200.0) / 12.0
                 df['y'] = -(df['y'] - 510.0) / 12.0
            
            # Attacking End Normalization
            last_frame = df['frame_idx'].max()
            end_data = df[(df['entity_type'] == 'player') & (df['frame_idx'] > last_frame - 50)]
            if not end_data.empty and end_data['x'].mean() < 0:
                df['x'] = -df['x']
                df['y'] = -df['y']
            
            # Filter to player
            df['entity_id'] = df['entity_id'].astype(str)
            my_track = df[df['entity_id'] == TARGET_PID]
            
            if my_track.empty: continue
            
            # Possession filter
            # Logic: If I don't have possession, I am off puck.
            # Simplified: Just grab all his points for now to see heatmap.
            # Or better: check possession events for that file.
            
            poss_events = infer_possession_events(df, threshold_ft=6.0)
            my_poss_frames = set()
            for _, pev in poss_events.iterrows():
                if str(pev['player_id']) == TARGET_PID:
                    for fr in range(int(pev['start_frame']), int(pev['end_frame'])+1):
                        my_poss_frames.add(fr)
            
            off_track = my_track[~my_track['frame_idx'].isin(my_poss_frames)]
            
            off_puck_x.extend(off_track['x'].tolist())
            off_puck_y.extend(off_track['y'].tolist())
            
            count += 1
            if count >= 10: break # He only has 10 goals, so this should get them all.
        except: pass

    if not off_puck_x:
        print("No data found.")
        return

    avg_x = np.mean(off_puck_x)
    avg_y = np.mean(off_puck_y)
    
    print(f"Stats for Grebenkin (n={count} goals):")
    print(f"Average Off-Puck Location: X={avg_x:.1f}, Y={avg_y:.1f}")
    print(f"  (X > 80 is deep slot/crease)")
    print(f"  (|Y| < 10 is central)")

if __name__ == "__main__":
    check_loc()
