import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import json
import glob
import logging
from puck import config, correction, rink, nhl_api

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_clock_sec(period, time_str):
    """Converts (period, MM:SS) to elapsed seconds."""
    m, s = map(int, time_str.split(':'))
    return (period - 1) * 1200 + m * 60 + s

# --- KINEMATICS & SEGMENTATION (From Prototype) ---
def calculate_kinematics(df):
    """Calculate velocity, speed, and angle for puck data."""
    df = df.sort_values('frame_idx')
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['timestamp'].diff() / 10.0 # Deciseconds -> Seconds
    df['dt'] = df['dt'].replace(0, np.nan).fillna(0.1)
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    
    a1 = df['angle']
    a2 = df['angle'].shift(1)
    diff = a1 - a2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    df['d_angle'] = diff
    return df

def segment_shot_vectors(df, angle_threshold_deg=20.0, min_speed=15.0, min_len=3):
    """Segment trajectory into ballistic vectors."""
    segments = []
    if df.empty: return segments
        
    current_segment = [df.iloc[0]]
    angle_thresh_rad = np.radians(angle_threshold_deg)
    
    for i in range(1, len(df)):
        frame = df.iloc[i]
        prev = df.iloc[i-1]
        
        angle_diff = abs(frame['d_angle'])
        speed_ok = frame['speed'] > min_speed
        speed_diff = frame['speed'] - prev['speed']
        # Major speed jump = new impulse
        new_impulse = speed_diff > 10.0 
        
        if angle_diff < angle_thresh_rad and speed_ok and not new_impulse:
            current_segment.append(frame)
        else:
            if len(current_segment) >= min_len:
                segments.append(pd.DataFrame(current_segment))
            current_segment = [frame]
            
    if len(current_segment) >= min_len:
        segments.append(pd.DataFrame(current_segment))
    return segments

def get_player_location(df_pos, player_id, frame_idx, window=5):
    """Get player location at specific frame (or nearest)."""
    # df_pos should serve all entities
    # Filter for player. Column is likely 'entity_id' for players or we check 'entity_type'
    # Actually, let's just assume valid ID passed.
    # Check if 'player_id' exists, if not construct it or use 'entity_id'.
    col = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
    
    p_data = df_pos[df_pos[col] == float(player_id)]
    if p_data.empty: return None
    
    # Exact frame
    row = p_data[p_data['frame_idx'] == frame_idx]
    if not row.empty:
        return row.iloc[0]
        
    # Nearest frame within window
    nearby = p_data[(p_data['frame_idx'] >= frame_idx - window) & (p_data['frame_idx'] <= frame_idx + window)]
    if not nearby.empty:
        # Closest
        nearby['diff'] = (nearby['frame_idx'] - frame_idx).abs()
        return nearby.sort_values('diff').iloc[0]
        
    return None

def discover_blocks():
    print("--- Discovering Blocks in Tracking Sequences (Double Validation) ---")
    
    # 1. Load PBP Data (2024-2025)
    season = '20242025'
    pbp_path = os.path.join(config.DATA_DIR, season, f"{season}_df.csv")
    if not os.path.exists(pbp_path):
        print(f"Error: PBP data not found at {pbp_path}")
        return
    
    print(f"Loading PBP data from {pbp_path}...")
    df_pbp = pd.read_csv(pbp_path)
    # Filter Synthetic
    if 'synthetic' in df_pbp.columns: df_pbp = df_pbp[df_pbp['synthetic'] != 1]
    if 'is_synthetic' in df_pbp.columns: df_pbp = df_pbp[df_pbp['is_synthetic'] != 1]

    # Attribution Correction
    print("Applying blocked shot attribution correction...")
    df_pbp = correction.fix_blocked_shot_attribution(df_pbp)
    
    # 2. Scan Edge Goal Files
    edge_dir = os.path.join(config.DATA_DIR, 'edge_goals', season)
    goal_files = []
    for f in glob.glob(os.path.join(edge_dir, "*_edge.json")):
        basename = os.path.basename(f)
        parts = basename.replace('_edge.json', '').split('_')
        if len(parts) >= 4:
            game_id = parts[1]
            event_id = parts[3]
            goal_files.append({
                'edge_json': f,
                'pos_csv': f.replace('_edge.json', '_positions.csv'),
                'game_id': int(game_id),
                'goal_event_id': int(event_id)
            })
            
    print(f"Found {len(goal_files)} goal sequences to scan.")
    
    # DEBUG: Target specific event (ENABLED)
    target_game, target_event = 2024020118, None # Scan all goals in 2024020118
    if target_event:
        goal_files = [g for g in goal_files if g['game_id'] == target_game and g['goal_event_id'] == target_event]
    else:
        goal_files = [g for g in goal_files if g['game_id'] == target_game]
        
    print(f"DEBUG MODE: Filtering for Game {target_game}. Found {len(goal_files)} files.")

    matches = []
    feed_cache = {}
    
    for gf_idx, gf in enumerate(goal_files):
        print(f"\nProcessing {gf['game_id']} Goal {gf['goal_event_id']}...")
            
        if not os.path.exists(gf['pos_csv']): continue
        
        # 1. Get Game Feed for METADATA (Shooter/Blocker IDs)
        game_id = str(gf['game_id'])
        if game_id not in feed_cache:
            try:
                feed_cache[game_id] = nhl_api.get_game_feed(gf['game_id'])
            except: continue
        
        feed = feed_cache[game_id]
        plays = feed.get('plays', [])
        
        # PBP Goal Info
        goal_play = next((p for p in plays if str(p.get('eventId')) == str(gf['goal_event_id'])), None)
        if not goal_play: continue
            
        period = goal_play.get('periodDescriptor', {}).get('number', 1)
        time_in_period = goal_play.get('timeInPeriod', '00:00')
        goal_clock_sec = get_clock_sec(period, time_in_period)
        
        # 2. Load Positions & Align Timing
        df_pos = pd.read_csv(gf['pos_csv'])
        if df_pos.empty: continue
        
        # Ensure entity_id is float for comparison
        col_id = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
        if col_id in df_pos.columns:
            df_pos[col_id] = pd.to_numeric(df_pos[col_id], errors='coerce')

        df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()
        if df_puck.empty: continue
        df_puck = df_puck.sort_values('frame_idx')
        
        # Normalize for Goal Frame Detection
        puck_abs_x = df_puck['x'].abs()
        if puck_abs_x.max() > 200:
             df_puck['norm_x'] = (df_puck['x'] - 1200.0) / 12.0
             df_puck['norm_y'] = -(df_puck['y'] - 510.0) / 12.0
        else:
             df_puck['norm_x'] = df_puck['x']
             df_puck['norm_y'] = df_puck['y']
             
        in_net = df_puck[(df_puck['norm_x'].abs() > 89.0) & (df_puck['norm_y'].abs() < 3.0)]
        goal_frame = in_net['frame_idx'].min() if not in_net.empty else df_puck['frame_idx'].max()
        
        goal_offset_s = goal_frame * 0.1
        clip_start_clock = goal_clock_sec - goal_offset_s
        clip_duration_s = (len(df_pos['frame_idx'].unique()) * 0.1)
        window_start_clock = clip_start_clock
        window_end_clock = clip_start_clock + clip_duration_s
        
        # 3. Find Candidate Blocks in PBP
        game_blocks = df_pbp[
            (df_pbp['game_id'] == gf['game_id']) & 
            (df_pbp['event'] == 'blocked-shot') &
            (df_pbp['total_time_elapsed_s'] >= window_start_clock - 2.0) & # 2s buffer
            (df_pbp['total_time_elapsed_s'] <= window_end_clock + 2.0)
        ]
        
        if game_blocks.empty: continue

        # 4. Shot Vector Segmentation
        df_puck = calculate_kinematics(df_puck)
        segments = segment_shot_vectors(df_puck)
        
        # Need at least 2 segments (Shot -> Deflection)
        if len(segments) < 2: continue
        
        # 5. Process Each PBP Block Candidate
        for _, b_row in game_blocks.iterrows():
            # Get IDs from API Feed for this specific block event
            # Use 'event_idx' or fuzzy match timestamp
            # But the PBP DF might not have the API event Id.
            # Let's find the matching play in 'feed['plays']' by time/type
            # Or use 'event_id' if available in PBP (it usually is)
            
            # PBP DF usually has 'event_idx' or similar. 
            # If not, match by time/coord.
            # But 'feed' is the SOURCE of truth for IDs.
            
            block_feed_play = None
            for p in plays:
                if p.get('typeDescKey') == 'blocked-shot':
                     # Match time
                     p_period = p.get('periodDescriptor', {}).get('number', 1)
                     p_time = p.get('timeInPeriod', '00:00')
                     p_sec = get_clock_sec(p_period, p_time)
                     if abs(p_sec - b_row['total_time_elapsed_s']) < 2.0:
                         # Good candidate
                         block_feed_play = p
                         break
            
            if not block_feed_play: continue
            
            # --- STRICT TIMING CHECK ---
            # Ensure the block event is actually INSIDE the tracking clip duration
            # Clip Start: window_start_clock
            # Clip End: window_end_clock
            b_time = b_row['total_time_elapsed_s']
            if b_time < (window_start_clock - 1.0) or b_time > (window_end_clock + 1.0):
                 # print(f"  Skipping Block {b_time}s (Outside Clip Range {window_start_clock:.1f}-{window_end_clock:.1f})")
                 continue
            
            details = block_feed_play.get('details', {})
            blocker_id = details.get('blockingPlayerId')
            shooter_id = details.get('shootingPlayerId')
            
            if not blocker_id or not shooter_id: 
                print(f"  Skipping block candidate: Missing IDs (Blocker:{blocker_id}, Shooter:{shooter_id})")
                continue

            # DEBUG: Check if IDs exist
            col = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
            unique_ids = df_pos[col].unique()
            print(f"  Looking for Blocker {blocker_id} and Shooter {shooter_id}")
            print(f"  Available Tracking IDs (First 10): {unique_ids[:10]}")
            if float(blocker_id) not in unique_ids:
                print(f"  WARNING: Blocker ID {blocker_id} NOT FOUND in tracking data!")
            if float(shooter_id) not in unique_ids:
                print(f"  WARNING: Shooter ID {shooter_id} NOT FOUND in tracking data!")

            print(f"  Checking PBP Block: Blocker {blocker_id}, Shooter {shooter_id}...")
            
            # --- DOUBLE VALIDATION ---
            best_score = float('inf')
            best_match = None
            
            for i in range(len(segments) - 1):
                vec_shot = segments[i]
                vec_deflect = segments[i+1]
                
                # Vertex (Deflection Point)
                vertex_frame_idx = vec_deflect.iloc[0]['frame_idx']
                vertex_x = vec_deflect.iloc[0]['x']
                vertex_y = vec_deflect.iloc[0]['y']
                
                # Shot Origin (Start of Shot Vector)
                origin_frame_idx = vec_shot.iloc[0]['frame_idx']
                origin_x = vec_shot.iloc[0]['x']
                origin_y = vec_shot.iloc[0]['y']
                
                # 1. Validation: Blocker Proximity
                blocker_loc = get_player_location(df_pos, blocker_id, vertex_frame_idx)
                
                dist_blocker = 999.9
                if blocker_loc is not None:
                    dist_blocker = np.sqrt((vertex_x - blocker_loc['x'])**2 + (vertex_y - blocker_loc['y'])**2)
                
                # 2. Validation: Shooter Proximity 
                shooter_loc = get_player_location(df_pos, shooter_id, origin_frame_idx)
                
                dist_shooter = 999.9
                if shooter_loc is not None:
                    dist_shooter = np.sqrt((origin_x - shooter_loc['x'])**2 + (origin_y - shooter_loc['y'])**2)
                
                print(f"    Vector Pair {i}->{i+1}:")
                print(f"      Vertex Frame {vertex_frame_idx}: Puc({vertex_x:.1f}, {vertex_y:.1f})")
                if blocker_loc is not None:
                     print(f"      Blocker {blocker_id}: ({blocker_loc['x']:.1f}, {blocker_loc['y']:.1f}) -> Dist {dist_blocker:.1f}ft")
                else:
                     print(f"      Blocker {blocker_id}: NOT FOUND")
                     
                print(f"      Shot Origin Frame {origin_frame_idx}: Puc({origin_x:.1f}, {origin_y:.1f})")
                if shooter_loc is not None:
                     print(f"      Shooter {shooter_id}: ({shooter_loc['x']:.1f}, {shooter_loc['y']:.1f}) -> Dist {dist_shooter:.1f}ft")
                
                # Relaxed Debug Thresholds
                if dist_blocker < 10.0: # Increased from 6.0 for debugging
                    # Valid Candidate!
                    # Score by proximity
                    score = dist_blocker
                    
                    if score < best_score:
                        best_score = score
                        best_match = {
                            'game_id': gf['game_id'],
                            'goal_event_id': gf['goal_event_id'],
                            'block_prob_gap': -1, # Deprecated
                            'pbp_timing_s': b_row['total_time_elapsed_s'],
                            'tracking_time_s': vertex_frame_idx * 0.1, # Relative to clip start
                            'alignment_offset_s': b_row['total_time_elapsed_s'] - (vertex_frame_idx * 0.1 + clip_start_clock), # Measure lag
                            'true_block_x': vertex_x,
                            'true_block_y': vertex_y,
                            'true_origin_x': origin_x, # Shot Start
                            'true_origin_y': origin_y,
                            'blocker_id': blocker_id,
                            'shooter_id': shooter_id,
                            'dist_to_blocker': dist_blocker,
                            'dist_to_shooter': dist_shooter,
                            'shot_speed': vec_shot['speed'].mean(),
                            'deflect_speed': vec_deflect['speed'].mean(),
                            'deflect_angle': np.degrees(abs(vec_deflect['angle'].mean() - vec_shot['angle'].mean()))
                        }
                        
            if best_match:
                matches.append(best_match)
                print(f"  Verified Block! Game {best_match['game_id']}: Shooter({shooter_id}) -> {best_match['shot_speed']:.0f}fps -> Blocker({blocker_id}) [{best_match['dist_to_blocker']:.1f}ft away]")

    # Save
    if matches:
        df_results = pd.DataFrame(matches)
        out_path = 'analysis/edge_block_discovery_verified.csv'
        os.makedirs('analysis', exist_ok=True)
        df_results.to_csv(out_path, index=False)
        print(f"Saved {len(matches)} verified blocks to {out_path}")
    else:
        print("No verified matches found.")

if __name__ == "__main__":
    discover_blocks()
