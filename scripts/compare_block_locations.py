import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import random
import pandas as pd
import numpy as np
import logging
import math
from puck import nhl_api
from puck import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_kinematics(df):
    """Calculate velocity, speed, and angle for puck data."""
    df = df.sort_values('frame_idx')
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    
    if 'timestamp' in df.columns:
        t_diff = df['timestamp'].diff().median()
        if t_diff > 0:
            if 0.9 <= t_diff <= 1.1:
                df['dt'] = 0.1 
            elif 90 <= t_diff <= 110:
                df['dt'] = 0.1
            else:
                 if t_diff < 0.2:
                      df['dt'] = t_diff
                 elif t_diff > 10.0:
                      df['dt'] = t_diff / 1000.0
                 else:
                      df['dt'] = 0.1
        else:
             df['dt'] = 0.1
    else:
        df['dt'] = 0.1 

    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    
    # Linearity
    angles_rad = np.radians(df['angle'].fillna(0).values)
    unwrapped = np.unwrap(angles_rad)
    df['angle_unwrapped'] = np.degrees(unwrapped)
    df['angle_std_linear'] = df['angle_unwrapped'].rolling(window=5, center=True, min_periods=1).std()
    
    return df

def process_file(pos_path, target_game_id, target_goal_id):
    """
    Process a single tracking file to find blocked shots and compare locations.
    Returns a list of result dicts.
    """
    results = []
    
    # 1. Load PBP
    try:
        feed = nhl_api.get_game_feed(target_game_id)
    except Exception as e:
        logger.error(f"Failed to load feed for {target_game_id}: {e}")
        return []

    # Find the GOAL event to establish time context
    goal_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(target_goal_id)), None)
    if not goal_play:
        return []
        
    # Get all BLOCKED SHOT events in this game
    # We want blocks that happened shortly before the goal (e.g. within 20 seconds)
    # Tracking data is usually short.
    # Note: PBP time is descending (Period Time Remaining) or ascending? usually 'timeInPeriod'
    
    goal_period = goal_play.get('periodDescriptor', {}).get('number')
    goal_time = goal_play.get('timeInPeriod') # "MM:SS"
    
    def parse_time(t_str):
        m, s = map(int, t_str.split(':'))
        return m * 60 + s
        
    goal_seconds = parse_time(goal_time)
    
    candidate_blocks = []
    for p in feed.get('plays', []):
        if p.get('typeDescKey') == 'blocked-shot':
            p_period = p.get('periodDescriptor', {}).get('number')
            if p_period == goal_period:
                p_time = p.get('timeInPeriod')
                p_seconds = parse_time(p_time)
                # Check within window (e.g. up to 30s before goal, or even slightly after if data aligns?)
                # Usually tracking file ends at goal.
                if 0 <= (goal_seconds - p_seconds) <= 30:
                    candidate_blocks.append(p)
                    
    if not candidate_blocks:
        return []
        
    # 2. Load Tracking Data
    try:
        df_pos = pd.read_csv(pos_path)
    except Exception:
        return []
        
    col_id = 'player_id' if 'player_id' in df_pos.columns else 'entity_id'
    if col_id in df_pos.columns:
        df_pos[col_id] = pd.to_numeric(df_pos[col_id], errors='coerce')
        
    # Puck
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()
    if df_puck.empty: return []
    
    # Normalize if needed (standard heuristic)
    if df_puck['x'].abs().max() > 200:
         def norm_x(x): return (x - 1200.0) / 12.0
         def norm_y(y): return -(y - 850.0) / 12.0 # Wait, Y center is 42.5ft? or 85ft width? Arena is 85 wide. Center 42.5. 
         # Standard NHL Edge raw: x=[0,2400]? y=[0,1000]? 
         # Usually x is -100 to 100.
         # Let's trust existing validation or just skip if crazy.
         # The previous script: (y - 510.0) / 12.0. 510 = 42.5 * 12.
         # So standard Y center is 0.
         pass # assume standard or handled by previous logic if copied EXACTLY.
         # Let's copy the norm logic from identification script exactly if needed.
         # "def norm_y(y): return -(y - 510.0) / 12.0"
         df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
         df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
         df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()

    df_puck = calculate_kinematics(df_puck)
    
    # 3. For each candidate block, try to find it in tracking
    for block in candidate_blocks:
        # PBP Info
        block_id = block.get('eventId')
        details = block.get('details', {})
        blocker_id = details.get('blockingPlayerId')
        shooter_id = details.get('shootingPlayerId')
        
        pbp_x = details.get('xCoord')
        pbp_y = details.get('yCoord')
        
        # If PBP coords missing, skip
        if pbp_x is None or pbp_y is None:
            continue
            
        # Match Blocker in Tracking
        # Find frames where puck is closest to blocker AND trajectory matches
        
        # Identify "Shot" candidates first (Ballistic high speed)
        # Using simplified logic from identify_shot_attempts
        high_speed_idx = df_puck[
            (df_puck['speed'] > 20.0) & 
            (df_puck['angle_std_linear'] < 40.0) 
        ].index
        
        if len(high_speed_idx) == 0: continue
        
        valid_frames = np.sort(df_puck.loc[high_speed_idx, 'frame_idx'].unique())
        # Group into segments
        if len(valid_frames) == 0: continue
        
        gaps = np.diff(valid_frames) > 2
        starts = [valid_frames[0]]
        starts.extend(valid_frames[1:][gaps])
        
        best_match_dist = 999.0
        best_track_x = None
        best_track_y = None
        
        # Check each shot candidate
        for s in starts:
            # Get shot vector
            row = df_puck[df_puck['frame_idx'] == s].iloc[0]
            
            # Project forward up to 30 frames (3s) to find Block Intersection
            min_b_dist = 999.9
            impact_x = 0
            impact_y = 0
            
            found_blocker = False
            
            for t in range(1, 30):
                f_check = s + t
                # Current Puck Pos (Projected or Actual? Actual is better if data exists)
                # But block stops the puck, so data might stop or go crazy.
                # Projecting is safer for "Expected Intersection".
                
                proj_x = row['x'] + (row['vx'] * 0.1 * t)
                proj_y = row['y'] + (row['vy'] * 0.1 * t)
                
                # Check Blocker Position
                if blocker_id:
                    df_b = df_pos[(df_pos[col_id] == float(blocker_id)) & (df_pos['frame_idx'] == f_check)]
                    if not df_b.empty:
                        bx, by = df_b.iloc[0]['x'], df_b.iloc[0]['y']
                        d = np.sqrt((proj_x - bx)**2 + (proj_y - by)**2)
                        
                        if d < min_b_dist:
                            min_b_dist = d
                            # The "Block Location" is where the puck hits the blocker
                            # So it's the BLOCKER's location (or Puck's projected location nearby)
                            # Let's use the Puck's Projected Location at minimum distance
                            # OR the Blocker's location?
                            # PBP records "Where the event happened".
                            impact_x = bx # Blocker's body location
                            impact_y = by
                            
            if min_b_dist < 5.0: # Valid block detection
                # Calculate Error vs PBP
                error = np.sqrt((impact_x - pbp_x)**2 + (impact_y - pbp_y)**2)
                
                if error < best_match_dist:
                    best_match_dist = error
                    best_track_x = impact_x
                    best_track_y = impact_y
                    
        if best_track_x is not None:
            results.append({
                'game_id': target_game_id,
                'block_event_id': block_id,
                'pbp_x': pbp_x,
                'pbp_y': pbp_y,
                'track_x': best_track_x,
                'track_y': best_track_y,
                'error': best_match_dist,
                'blocker_id': blocker_id
            })
            
    return results

def main():
    print("Searching for files...")
    # Get all positions.csv files
    files = glob.glob(os.path.join(config.DATA_DIR, 'edge_goals', '20242025', '*_positions.csv'))
    
    if not files:
        print("No files found.")
        return
        
    print(f"Found {len(files)} files. Sampling 20...")
    sample_files = random.sample(files, min(len(files), 100)) # Try 100? No, PBP load is slow. 20 is safe.
    
    all_results = []
    
    for f in sample_files:
        try:
            # Parse filename: game_2024020137_goal_1040_positions.csv
            basename = os.path.basename(f)
            parts = basename.split('_')
            # game, {id}, goal, {id}, positions.csv
            if len(parts) >= 5:
                game_id = parts[1]
                goal_id = parts[3]
                
                print(f"Processing {game_id} Goal {goal_id}...")
                res = process_file(f, game_id, goal_id)
                if res:
                    print(f"  Found {len(res)} matches.")
                    all_results.extend(res)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not all_results:
        print("No paired blocked shots found.")
        return
        
    df_res = pd.DataFrame(all_results)
    print("\n--- RESULTS ---")
    print(df_res.describe())
    
    out_file = 'analysis/blocked_shot_comparison.csv'
    os.makedirs('analysis', exist_ok=True)
    df_res.to_csv(out_file, index=False)
    print(f"Saved to {out_file}")
    
    # Calculate Mean Error
    mean_err = df_res['error'].mean()
    median_err = df_res['error'].median()
    print(f"Mean Error: {mean_err:.2f} ft")
    print(f"Median Error: {median_err:.2f} ft")

if __name__ == "__main__":
    main()
