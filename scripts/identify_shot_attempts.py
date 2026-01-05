import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import logging
from puck import nhl_api
from puck import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_kinematics(df):
    """Calculate velocity, speed, and angle for puck data."""
    df = df.sort_values('frame_idx')
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = 0.1 # Fixed 10Hz
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    
    # Circular Standard Deviation Proxy
    # Calculate difference between consecutive angles, handling wrap
    diff = np.abs(df['angle'].diff())
    diff = np.minimum(diff, 360 - diff)
    
    # "Linearity" = Rolling mean of this difference
    # Low value means stable direction
    df['angle_std'] = diff.rolling(window=3, center=True).mean().fillna(0)
    
    return df

def identify_shots():
    # Parse Args
    if len(sys.argv) > 2:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
        target_block_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        # Default to previous test case
        target_game_id = 2024020202
        target_goal_id = 328
        target_block_id = 325
    
    print(f"--- Identifying Shot Attempts for Game {target_game_id} Goal {target_goal_id} ---")

    # 1. Load Data
    season = '20242025'
    pos_path = f"data/edge_goals/{season}/game_{target_game_id}_goal_{target_goal_id}_positions.csv"
    
    if not os.path.exists(pos_path):
        print("Data not found.")
        return
        
    df_pos = pd.read_csv(pos_path)
    # Cast IDs
    col_id = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
    if col_id in df_pos.columns:
        df_pos[col_id] = pd.to_numeric(df_pos[col_id], errors='coerce')

    # Puck Data
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()
    if df_puck.empty: return
    
    # Debug
    print(f"DEBUG: Max Abs X: {df_puck['x'].abs().max()}")
    f8 = df_puck[df_puck['frame_idx'] == 8]
    if not f8.empty:
        print(f"DEBUG: Raw Frame 8 Y: {f8['y'].iloc[0]}")
    
    # Coordinate Normalization (if needed)
    # FORCE DISABLED: Data is already normalized.
    # if df_puck['x'].abs().max() > 200:
    #      print("DEBUG: Normalizing...")
    #      df_puck['x'] = (df_puck['x'] - 1200.0) / 12.0
    #      df_puck['y'] = -(df_puck['y'] - 510.0) / 12.0
    #      df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
    #      df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0

    df_puck = calculate_kinematics(df_puck)
    
    # 2. Get Metadata (Shooter ID)
    feed = nhl_api.get_game_feed(target_game_id)
    # We want the BLOCK event to find the shooter (or the Goal event? The Block event has the shooter!)
    # Let's check BOTH Block (325) and Goal (328) for the shooter ID.
    
    # block_id passed as arg
    if target_block_id:
        block_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(target_block_id)), None)
    else:
        block_play = None
    
    shooter_id = None
    if block_play:
        shooter_id = block_play.get('details', {}).get('shootingPlayerId')
        print(f"PBP Block Event {target_block_id} Shooter ID: {shooter_id}")
        
    if not shooter_id:
        # Fallback to Goal event scorer?
        goal_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(target_goal_id)), None)
        if goal_play:
            shooter_id = goal_play.get('details', {}).get('scoringPlayerId')
            print(f"Fallback to Goal Scorer ID: {shooter_id}")
            
    if not shooter_id:
        print("Could not identify Shooter ID.")
        return
        
    print(f"Target Shooter ID: {shooter_id}")
    
    # 3. Find Candidate Shot Definitions
    # Logic: High Speed (>30fps) Start of Vector + Proximity to Shooter (<3ft)
    
    # Segment vectors simply by speed threshold first
    # Find frames where speed JUMPS from Low to High
    
    SHOT_SPEED_THRESH = 40.0 # ft/s (approx 27mph - weak shot but decent threshold)
    
    # Calculate Linearity (Rolling Angle Std)
    df_puck['angle_std'] = df_puck['angle'].rolling(window=5, center=True).std()
    
    # Filter for ballistic frames (Speed > 10 AND Angle Std < 30)
    # We want to catch EVERYTHING in the CSV for debugging
    high_speed = df_puck[(df_puck['speed'] > 10.0) & (df_puck['angle_std'] < 30.0)] 
    
    if high_speed.empty:
        print("No high speed linear puck detected.")
        return
        
    # Group into continuous segments
    high_speed['group'] = (high_speed['frame_idx'].diff() > 1).cumsum()
    
    candidates = []
    
    # --- 3. CANDIDATE GENERATION & SCORING ---
    candidates = []
    
    # Iterate through potential start frames
    # We scan linear segments or simply search high-activity frames?
    # Let's Scan all linear-ish segments relative to net
    
    # Parameters for Scoring
    # User emphasized "vector looks like a shot" (High Speed is #1 factor)
    W_SPEED = 0.7      # Critical: Projectile motion
    W_PROXIMITY = 0.15 # Important but secondary to kinematics
    W_ALIGNMENT = 0.15 # Must be aimed at net
    
    # Filter: Minimal criteria to be considered "Shot-Like"
    # Relaxed for Debugging Scope
    # Speed > 10 fps (was 20/40)
    # Linearity < 30 deg (was 10/15)
    
    segment_starts = df_puck[
        (df_puck['speed'] > 10.0) & 
        (df_puck['angle_std'] < 30.0) 
    ].index
    
    # We want unique 'starts' of segments.
    # Let's map indices back to frames
    valid_frames = df_puck.loc[segment_starts, 'frame_idx'].unique()
    
    # Group consecutive frames
    if len(valid_frames) > 0:
        valid_frames = np.sort(valid_frames)
        gaps = np.diff(valid_frames) > 1
        # Add first element
        starts = [valid_frames[0]]
        starts.extend(valid_frames[1:][gaps])
    else:
        starts = []
        
    # BACK-TRACE REFINEMENT
    # Rolling window (Standard Deviation) often disqualifies the very first frames of a shot
    # because the window overlaps with pre-shot chaos.
    # We trace back from the "Detected Start" to find the "True Release" (first high-speed frame).
    refined_starts = []
    
    for s in starts:
        current_start = s
        # Look back up to 5 frames
        for i in range(1, 6):
            prev_f = s - i
            prev_row = df_puck[df_puck['frame_idx'] == prev_f]
            if prev_row.empty: break
            
            # If previous frame is also high speed (> 40 fps), it's part of the shot
            spd = prev_row['speed'].iloc[0]
            # print(f"DEBUG: Back-Trace F{s} -> F{prev_f}: Speed {spd:.1f}")
            if spd > 40.0:
                current_start = prev_f
            else:
                # Speed dropped. Stop at current_start (which is the last high-speed frame).
                break 
        
        # PROXIMITY REFINEMENT (Window: Start to Start+2)
        # Scan forward slightly to allow for tracking noise, 
        # but ensure we don't drift too far from the high-speed start.
        best_start = current_start
        min_dist = 999.9
        
        for f_cand in range(current_start, current_start + 3):
             df_s = df_pos[(df_pos[col_id] == float(shooter_id)) & (df_pos['frame_idx'] == f_cand)]
             df_p = df_puck[df_puck['frame_idx'] == f_cand]
             
             if not df_s.empty and not df_p.empty:
                 sx, sy = df_s.iloc[0]['x'], df_s.iloc[0]['y']
                 px, py = df_p.iloc[0]['x'], df_p.iloc[0]['y']
                 d = np.sqrt((px-sx)**2 + (py-sy)**2)
                 
                 # Only update if significantly better? 
                 # Or just minimize.
                 # If we are strictly ballistic, min_dist might be at current_start or +1.
                 if d < min_dist:
                     min_dist = d
                     best_start = f_cand
        
        current_start = best_start
        
        if current_start not in refined_starts:
            refined_starts.append(current_start)
            
    starts = refined_starts
        
    # Determine Net Direction
    mean_vx = df_puck['vx'].mean()
    net_x = -89.0 if mean_vx < 0 else 89.0
    print(f"Detected Attack Direction: Net at X={net_x}")
    
    print(f"\nEvaluating {len(starts)} Candidate Segments...")
    
    # --- REFINEMENT STEP: FORWARD SCAN ---
    # Skip "weak" launch frames if they are too slow (<40fps)
    refined_final_starts = []
    for s in starts:
        current_start = s
        row = df_puck[df_puck['frame_idx'] == current_start]
        
        # If weak start, look ahead
        if not row.empty and row['speed'].iloc[0] < 40.0:
            for i in range(1, 6):
                next_f = s + i
                next_row = df_puck[df_puck['frame_idx'] == next_f]
                if not next_row.empty and next_row['speed'].iloc[0] > 40.0:
                    current_start = next_f
                    break
        
        if current_start not in refined_final_starts:
            refined_final_starts.append(current_start)
            
    starts = refined_final_starts

    # --- MAIN SCORING LOOP ---
    for start_frame in starts:
        row = df_puck[df_puck['frame_idx'] == start_frame].iloc[0]
        
        # Handling Launch Frames (Low Speed Start)
        effective_speed = row['speed']
        is_launch_frame = False
        
        if effective_speed < 40.0:
            # Check next frame
            next_row = df_puck[df_puck['frame_idx'] == start_frame + 1]
            if not next_row.empty and next_row['speed'].iloc[0] > 40.0:
                effective_speed = next_row['speed'].iloc[0] # Borrow speed from shot
                is_launch_frame = True
        
        print(f"Checking Frame {start_frame}: Speed {effective_speed:.1f} (Launch: {is_launch_frame})")
        
        # 1. Alignment (Net Vector)
        dx_net = net_x - row['x']
        dy_net = 0 - row['y']
        angle_to_net = np.degrees(np.arctan2(dy_net, dx_net))
        
        dev = abs(row['angle'] - angle_to_net)
        dev = dev % 360
        if dev > 180: dev = 360 - dev
        
        # Relax deviation check for Launch frames (angle might be weird during acceleration)
        if not is_launch_frame and dev > 60: continue 
        
        # 2. Proximity
        df_shooter = df_pos[(df_pos[col_id] == float(shooter_id)) & (df_pos['frame_idx'] == start_frame)]
        dist = 99.9
        if not df_shooter.empty:
            sx, sy = df_shooter.iloc[0]['x'], df_shooter.iloc[0]['y']
            px, py = row['x'], row['y']
            dist = np.sqrt((px-sx)**2 + (py-sy)**2)
            
        print(f"  Frame {start_frame} Dist: {dist:.1f}")
            
        if dist > 25.0 and effective_speed < 50.0: 
            print(f"  -> Dropped F{start_frame}: Dist {dist:.1f} and Low Speed")
            continue 
        
        # 3. Scoring
        # Revised Logic: Distance is a GATE (Multiplicative).
        # A 100mph shot 50ft from the shooter is a PASS or a DEFLECTION, not the release.
        
        s_speed = min(max((effective_speed - 20) / 80, 0), 1.0) # 20-100 fps mapping
        
        # Alignment: 0-45 deg -> 1.0-0.0
        s_align = max(1 - (dev / 45), 0)
        
        # Proximity Gate: 
        # < 5ft: 1.0
        # 5-25ft: Linear decay to 0.0
        # > 25ft: 0.0
        if dist <= 5.0:
            s_dist = 1.0
        else:
            s_dist = max(1 - ((dist - 5.0) / 20.0), 0)
            
        # Composite Score
        # Speed and Alignment define the "Quality" of the vector
        # Proximity confirms it is "Yours" (The Shooter's)
        base_quality = (0.7 * s_speed) + (0.3 * s_align)
        score = base_quality * s_dist
        
        print(f"  -> Candidate F{start_frame}: Score {score:.3f} (Q:{base_quality:.2f} DistGate:{s_dist:.2f})")
        
        candidates.append({
            'frame_idx': start_frame,
            'score': score,
            'speed': effective_speed, # Store effective speed
            'dist': dist,
            'dev_deg': dev,
            'x': row['x'], 'y': row['y'],
            'vx': row['vx'], 'vy': row['vy']
        })
        
    # --- 4. SELECTION ---
    if not candidates:
        print("[FAILURE] No valid shot candidates found.")
        return

    # Sort by Score Descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nTop Candidates:")
    for c in candidates:
        print(f"  F{int(c['frame_idx'])} | Score: {c['score']:.3f} | Spd: {c['speed']:.1f} | Dist: {c['dist']:.1f}ft | Dev: {c['dev_deg']:.1f}°")
        
    best = candidates[0]
    print(f"\n[SUCCESS] Identified Best Shot Candidate: Frame {best['frame_idx']}")
    print(f"  Confidence Score: {best['score']:.3f}")
    print(f"  Location: ({best['x']:.1f}, {best['y']:.1f})")
    print(f"  Speed: {best['speed']:.1f} fps")
    print(f"  Shooter Proximity: {best['dist']:.1f} ft")
    print(f"  Net Alignment: {best['dev_deg']:.1f} deg deviation")

    # Export Candidates for Visualization
    out_dir = os.path.join(config.DATA_DIR, '..', 'analysis', 'blocked_shots') # Ensure analysis/blocked_shots
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'candidate_vectors_{target_game_id}_{target_goal_id}_v2.csv')
    pd.DataFrame(candidates).to_csv(out_csv, index=False)
    print(f"Candidates saved to {out_csv}")

if __name__ == "__main__":
    identify_shots()
