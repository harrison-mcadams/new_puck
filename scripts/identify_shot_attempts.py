import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import logging
from puck import nhl_api
from puck import nhl_api
from puck import config
import requests

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

def get_goalie_ids(game_id):
    """Fetches goalie IDs for the game to exclude them from blocker checks."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    goalie_ids = []
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Home
            for g in data.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', []):
                goalie_ids.append(float(g.get('playerId')))
            # Away
            for g in data.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', []):
                goalie_ids.append(float(g.get('playerId')))
    except Exception as e:
        print(f"[WARN] Failed to fetch goalies: {e}")
    return set(goalie_ids)

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
    if df_puck['x'].abs().max() > 200:
         print("DEBUG: Normalizing coordinates...")
         def norm_x(x): return (x - 1200.0) / 12.0
         def norm_y(y): return -(y - 510.0) / 12.0
         
         df_pos['x'] = norm_x(df_pos['x'])
         df_pos['y'] = norm_y(df_pos['y'])
         
         # Re-extract normalized puck/players
         df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()

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
    
    # Get Blocker ID
    blocker_id = None
    if block_play:
        blocker_id = block_play.get('details', {}).get('blockingPlayerId')
        print(f"Target Blocker ID: {blocker_id}")
    
    # 3. Find Candidate Shot Definitions
    # Logic: High Speed (>30fps) Start of Vector + Proximity to Shooter (<3ft)
    
    # Calculate Linearity (Rolling Angle Std)
    # Use unwrap to handle -180/180 crossing smoothly
    angles_rad = np.radians(df_puck['angle'].fillna(0).values)
    unwrapped = np.unwrap(angles_rad)
    df_puck['angle_unwrapped'] = np.degrees(unwrapped)
    df_puck['angle_unwrapped'] = np.degrees(unwrapped)
    df_puck['angle_std_linear'] = df_puck['angle_unwrapped'].rolling(window=5, center=True, min_periods=1).std()
    
    # Filter for ballistic frames (Speed > 10 AND Angle Std < 35)
    # Relaxing angle std slightly to 35 for pre-season/noisy data
    high_speed_idx = df_puck[
        (df_puck['speed'] > 10.0) & 
        (df_puck['angle_std_linear'] < 35.0) 
    ].index
    
    # NEW: Acceleration-based Burst Detection
    # Catch shots blocked immediately (within 2-3 frames) which fail 5-frame linearity
    df_puck['accel'] = df_puck['speed'].diff()
    burst_idx = df_puck[
        (df_puck['speed'] > 15.0) &
        (df_puck['accel'] > 10.0) # explosive start
    ].index
    
    # Combine
    combined_idx = high_speed_idx.union(burst_idx)

    # 4. Group into Shot Events (consecutive frames)
    # If gap > 1 frame, it's a new shot
    if len(combined_idx) == 0:
        print("No high-speed ballistic frames found.")
        return

    valid_frames = np.sort(df_puck.loc[combined_idx, 'frame_idx'].unique())
    gaps = np.diff(valid_frames) > 1
    starts = [valid_frames[0]]
    starts.extend(valid_frames[1:][gaps])
    print(f"DEBUG STARTS: {starts}")
        
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
             # Try Intended Shooter
             dist_to_cand = 999.9
             if shooter_id:
                 df_s = df_pos[(df_pos[col_id] == float(shooter_id)) & (df_pos['frame_idx'] == f_cand)]
                 df_p = df_puck[df_puck['frame_idx'] == f_cand]
                 if not df_s.empty and not df_p.empty:
                     sx, sy = df_s.iloc[0]['x'], df_s.iloc[0]['y']
                     px, py = df_p.iloc[0]['x'], df_p.iloc[0]['y']
                     dist_to_cand = np.sqrt((px-sx)**2 + (py-sy)**2)
             
             # Fallback if shooter missing or too far
             if dist_to_cand > 20.0:
                 df_frame = df_pos[(df_pos['frame_idx'] == f_cand) & (df_pos['entity_type'] == 'player')]
                 df_p = df_puck[df_puck['frame_idx'] == f_cand]
                 if not df_frame.empty and not df_p.empty:
                    px, py = df_p.iloc[0]['x'], df_p.iloc[0]['y']
                    dists = np.sqrt((df_frame['x'] - px)**2 + (df_frame['y'] - py)**2)
                    dist_to_cand = dists.min()

             if dist_to_cand < min_dist:
                 # If we already have a "good" proximity (< 5ft) at an EARLIER frame,
                 # don't pull forward to a later frame just because an opponent (fallback) 
                 # or tracking noise is "closer". 
                 # Exception: If the current min_dist is from fallback and we find the INTENDED shooter, we might update.
                 # But generally, we want the RELEASE frame.
                 if min_dist < 5.0:
                     continue
                 
                 min_dist = dist_to_cand
                 best_start = f_cand
        
        current_start = best_start
        
        if current_start not in refined_starts:
            refined_starts.append(current_start)
            
    starts = refined_starts
        
    # Determine Net Direction
    # Use velocity of candidate shot frames, NOT global mean
    candidate_vx = df_puck.loc[combined_idx, 'vx'].mean()
    net_x = -89.0 if candidate_vx < 0 else 89.0
    print(f"Detected Attack Direction (from candidates): Net at X={net_x}")
    
    print(f"\nEvaluating {len(starts)} Candidate Segments...")
    
    candidates = []
    
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

    # Get Goalies to exclude
    goalie_ids = get_goalie_ids(target_game_id)
    print(f"Goalies to exclude: {goalie_ids}")

    # --- MAIN SCORING LOOP ---
    for start_frame in starts:
        start_frame = int(start_frame)
        print(f"Checking Frame {start_frame}...")
        
        df_f = df_puck[df_puck['frame_idx'] == start_frame]
        if df_f.empty:
            print(f"  -> WARNING: Frame {start_frame} missing from puck data.")
            continue
        row = df_f.iloc[0]
        
        # Handling Launch Frames (Low Speed Start)
        effective_speed = row['speed']
        is_launch_frame = False
        
        if effective_speed < 40.0:
            # Check next frame
            next_row = df_puck[df_puck['frame_idx'] == start_frame + 1]
            if not next_row.empty and next_row['speed'].iloc[0] > 40.0:
                effective_speed = next_row['speed'].iloc[0] # Borrow speed from shot
                is_launch_frame = True
        
        # 1. Alignment (Net Vector)
        # Check BOTH nets since attack direction might be ambiguous
        nets = [-89.0, 89.0]
        devs = []
        for nx in nets:
            dx_net = nx - row['x']
            dy_net = 0 - row['y'] # standard net y=0
            angle_to_net = np.degrees(np.arctan2(dy_net, dx_net))
            d = abs(row['angle'] - angle_to_net)
            d = d % 360
            if d > 180: d = 360 - d
            devs.append(d)
            
        dev = min(devs)
        
        # Relax deviation check for Launch frames (angle might be weird during acceleration)
        if not is_launch_frame and dev > 60: continue 
        
        # 2. Proximity
        dist = 99.9
        if shooter_id:
            df_shooter = df_pos[(df_pos[col_id] == float(shooter_id)) & (df_pos['frame_idx'] == start_frame)]
            if not df_shooter.empty:
                sx, sy = df_shooter.iloc[0]['x'], df_shooter.iloc[0]['y']
                px, py = row['x'], row['y']
                dist = np.sqrt((px-sx)**2 + (py-sy)**2)
        
        # Strict Shooter Check: No fallback
        if dist > 30.0:
            pass # Previously fallback to closest player. Now ignored.
            
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
        
        if dist <= 5.0:
            s_dist = 1.0
        else:
            s_dist = max(1 - ((dist - 5.0) / 20.0), 0)
        
        # 4. Blocker Plausibility Check (Forward Trace)
        min_blocker_dist = 99.9
        min_any_player_dist = 99.9 # Generic Closest Player
        any_player_id = None
        
        available_ids = df_pos[col_id].unique()
        
        blocker_in_data = False
        if blocker_id and float(blocker_id) in available_ids:
            blocker_in_data = True
            
        for t in range(1, 40): # Look ahead 4 seconds max
            f_check = start_frame + t
            
            proj_x = row['x'] + (row['vx'] * 0.1 * t)
            proj_y = row['y'] + (row['vy'] * 0.1 * t)

            # Specific Blocker Check
            if blocker_in_data:
                df_b = df_pos[(df_pos[col_id] == float(blocker_id)) & (df_pos['frame_idx'] == f_check)]
                if not df_b.empty:
                    bx, by = df_b.iloc[0]['x'], df_b.iloc[0]['y']
                    d_b = np.sqrt((proj_x - bx)**2 + (proj_y - by)**2)
                    if d_b < min_blocker_dist:
                        min_blocker_dist = d_b
            
            # Generic Player Check Removed (Strict Verification)
            pass

        if blocker_in_data:
            blocker_dist = min_blocker_dist
            print(f"  -> Blocker Check: Min Dist {blocker_dist:.1f}ft to ID {blocker_id}")
        else:
             print(f"  -> Blocker Check: ID {blocker_id} not found in tracking (Strict).")

        # Blocker Gate:
        s_blocker = 1.0
        if blocker_in_data:
            blocker_dist = min_blocker_dist
            if blocker_dist < 6.0: s_blocker = 1.0 
            elif blocker_dist > 20.0: s_blocker = 0.0 
            else: s_blocker = max(1 - ((blocker_dist - 6.0) / 14.0), 0) 
        elif blocker_id:
            # Strict Penalty for Missing Blocker
            s_blocker = 0.1

            
        
        # Composite Score
        # Speed and Alignment define the "Quality" of the vector
        # Proximity confirms it is "Yours" (The Shooter's)
        # Blocker Dist confirms it "Intersects" (The Block)
        
        base_quality = (0.7 * s_speed) + (0.3 * s_align)
        score = base_quality * s_dist * s_blocker
        
        print(f"  -> Candidate F{start_frame}: Score {score:.3f} (Q:{base_quality:.2f} DistGate:{s_dist:.2f})")
        
        candidates.append({
            'frame_idx': start_frame,
            'score': score,
            'speed': effective_speed, # Store effective speed
            'dist': dist,
            'blocker_dist': min_blocker_dist,
            'blocker_id': blocker_id,
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
        print(f"  F{int(c['frame_idx'])} | Score: {c['score']:.3f} | Spd: {c['speed']:.1f} | Dist: {c['dist']:.1f}ft | BlockDist: {c['blocker_dist']:.1f}ft | Dev: {c['dev_deg']:.1f}°")
        
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
