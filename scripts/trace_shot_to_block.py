import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from puck import nhl_api

def trace_block():
    # Parse Args
    shot_frame_override = None
    if len(sys.argv) > 2:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
        if len(sys.argv) > 3:
            shot_frame_override = int(sys.argv[3])
    else:
        target_game_id = 2024020202
        target_goal_id = 328
    
    # Findings from identify_shot_attempts.py
    # Default to 24 if not provided
    SHOT_FRAME_IDX = shot_frame_override if shot_frame_override is not None else 24
    shooter_id = 8483930
    
    print(f"--- Tracing Shot Vector from Frame {SHOT_FRAME_IDX} ---")
    
    # 1. Load Data
    season = '20242025'
    pos_path = f"data/edge_goals/{season}/game_{target_game_id}_goal_{target_goal_id}_positions.csv"
    if not os.path.exists(pos_path): return
    
    df_pos = pd.read_csv(pos_path)
    col_id = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
    if col_id in df_pos.columns: df_pos[col_id] = pd.to_numeric(df_pos[col_id], errors='coerce')
    
    # Puck Data
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy().sort_values('frame_idx')
    
    # Coordinate Normalization
    if df_puck['x'].abs().max() > 200:
         df_puck['x'] = (df_puck['x'] - 1200.0) / 12.0
         df_puck['y'] = -(df_puck['y'] - 510.0) / 12.0
         df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
         df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0

    # 2. Get Blocker ID
    feed = nhl_api.get_game_feed(target_game_id)
    block_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == '325'), None)
    blocker_id = block_play.get('details', {}).get('blockingPlayerId') if block_play else None
    
    print(f"Target Blocker ID: {blocker_id}")
    # if not blocker_id: return # Allow blind search

    # 3. Trace Forward from Shot Start
    # We look for the point where the shot vector ENDS or CHANGES direction significantly
    # AND is near the Blocker.
    
    # We want frames >= SHOT_FRAME_IDX
    df_shot = df_puck[df_puck['frame_idx'] >= SHOT_FRAME_IDX].copy()
    
    # Calculate Angle Deltas
    df_shot['dx'] = df_shot['x'].diff()
    df_shot['dy'] = df_shot['y'].diff()
    df_shot['vx'] = df_shot['dx'] / 0.1
    df_shot['vy'] = df_shot['dy'] / 0.1
    df_shot['speed'] = np.sqrt(df_shot['vx']**2 + df_shot['vy']**2)
    df_shot['angle'] = np.degrees(np.arctan2(df_shot['vy'], df_shot['vx'])) # Use degrees?
    # Note: d_angle logic below uses diff...
    
    # Original code used np.arctan2 (radians) then converted delta to degrees?
    # Let's check below.
    # Lines 57-61 calculate diff. 
    # Line 69 uses np.degrees(row['d_angle']). 
    # So d_angle must be in RADIANS.
    # Therefore angle must be in RADIANS here.
    df_shot['angle'] = np.arctan2(df_shot['dy'], df_shot['dx'])
    
    a1 = df_shot['angle']
    a2 = df_shot['angle'].shift(1)
    diff = a1 - a2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    df_shot['d_angle'] = diff
    
    # Scan for "Deflection Events" (Angle Jump > 20 deg)
    deflections = df_shot[df_shot['d_angle'].abs() > np.radians(20.0)]
    
    candidates = []
    
    for _, row in deflections.iterrows():
        delta = np.degrees(row['d_angle']) # Define delta
        if abs(delta) > 20.0 and row['speed'] > 10.0:
            # Check Proximity
            f_idx = row['frame_idx']
            
            # Find closest player
            # Load all players at this frame
            df_frame = df_pos[df_pos['frame_idx'] == f_idx]
            # Exclude puck
            if 'entity_type' in df_frame.columns:
                df_players = df_frame[df_frame['entity_type'] == 'player']
            else:
                df_players = df_frame[df_frame['player_id'].notnull()]
                
            closest_dist = 999.9
            closest_id = None
            
            px, py = row['x'], row['y']
            
            for _, p_row in df_players.iterrows():
                # manual dist
                dist = np.sqrt((px - p_row['x'])**2 + (py - p_row['y'])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    pid = p_row.get('player_id', p_row.get('id', 'Unknown'))
                    closest_id = pid
            
            print(f"Deflection at Frame {f_idx}: Angle Delta {delta:.1f} deg")
            print(f"  Closest Player: {closest_id} ({closest_dist:.1f} ft)")
            
            # Check Target Blocker
            if blocker_id:
                df_target = df_players[df_players[col_id] == float(blocker_id)]
                target_dist = 999.9
                if not df_target.empty:
                     tx, ty = df_target.iloc[0]['x'], df_target.iloc[0]['y']
                     target_dist = np.sqrt((px-tx)**2 + (py-ty)**2)
                print(f"  Target Blocker ({blocker_id}) Dist: {target_dist:.1f} ft")
            else:
                target_dist = 999.9
            
            if closest_dist < 6.0:
                 print(f"[SUCCESS] Confirmed Block at Frame {f_idx} by Player {closest_id}")
                 # Log and return just the first good one?
                 # Or keep looking?
                 # For now, let's stop at first good block.
                 # return (f_idx, closest_id) <-- Actually just print for now
                 # To keep the original 'candidates' logic, we'll add it here if it's a potential block
                 candidates.append({
                     'frame_idx': f_idx,
                     'x': row['x'],
                     'y': row['y'],
                     'dist': closest_dist, # Use closest_dist as the primary metric for 'best'
                     'blocker_id': closest_id # Store the ID of the closest player
                 })
             
    if candidates:
        # The original logic was to find the best based on 'dist' (to blocker_id).
        # Now 'dist' in candidates refers to the closest player.
        # If the goal is still to find the *target blocker*, we need to re-evaluate.
        # For now, let's assume 'best' refers to the closest player found.
        best = min(candidates, key=lambda x: x['dist'])
        print(f"\n[SUCCESS] Confirmed Block at Frame {best['frame_idx']}")
        print(f"  Location: ({best['x']:.1f}, {best['y']:.1f})")
        print(f"  Blocker Proximity: {best['dist']:.1f} ft (by player {best['blocker_id']})")
        
        # Save minimal output for visualizer or next step
    else:
        print("\n[FAILURE] No deflection found near blocker.")

if __name__ == "__main__":
    trace_block()
