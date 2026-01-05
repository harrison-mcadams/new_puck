import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from puck import nhl_api

def debug_timing():
    game_id = 2024020202
    goal_event_id = 328
    
    print(f"--- Debugging Timing for Game {game_id} Goal {goal_event_id} ---")
    
    # 1. Fetch API Feed for PBP Timestamps
    feed = nhl_api.get_game_feed(game_id)
    if not feed:
        print("Error: Could not fetch game feed.")
        return

    # Find Goal
    goal_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(goal_event_id)), None)
    if not goal_play:
        print("Error: Goal event not found in API.")
        return
        
    goal_time_str = goal_play.get('timeInPeriod')
    goal_period = goal_play.get('periodDescriptor', {}).get('number')
    
    def get_abs_sec(period, t_str):
        m, s = map(int, t_str.split(':'))
        return (period - 1) * 1200 + m * 60 + s
        
    goal_sec = get_abs_sec(goal_period, goal_time_str)
    print(f"Goal Logic: Period {goal_period}, Time {goal_time_str} ({goal_sec}s)")
    
    # Find preceding blocks (last 30 seconds)
    blocks = []
    for p in feed.get('plays', []):
        if p.get('typeDescKey') == 'blocked-shot':
            p_per = p.get('periodDescriptor', {}).get('number')
            p_time = p.get('timeInPeriod')
            p_sec = get_abs_sec(p_per, p_time)
            
            if p_sec <= goal_sec and (goal_sec - p_sec) < 30:
                blocks.append({
                    'id': p.get('eventId'),
                    'sec': p_sec,
                    'time': p_time,
                    'blocker': p.get('details', {}).get('blockingPlayerId'),
                    'shooter': p.get('details', {}).get('shootingPlayerId')
                })
                
    print(f"Found {len(blocks)} candidate blocks in 30s window preceding goal:")
    for b in blocks:
        diff = goal_sec - b['sec']
        print(f"  - Block {b['id']} @ {b['time']} ({diff}s before goal). Blocker {b['blocker']}")

    # 2. Check Tracking Data Range
    season = '20242025'
    pos_path = f"data/edge_goals/{season}/game_{game_id}_goal_{goal_event_id}_positions.csv"
    
    if not os.path.exists(pos_path):
        print("Tracking CSV not found.")
        return
        
    df = pd.read_csv(pos_path)
    
    # In tracking, we often don't have absolute game clock, only frame_idx is reliable relative time.
    # But usually there is a 'game_clock' or similar column? 
    # Let's check columns.
    print(f"\nTracking Columns: {list(df.columns)}")
    
    # Assuming standard Edge: usually 10 frames = 1 sec.
    # We find the goal frame by the 'goal' event in tracking? No, usually inferred.
    # Let's find when the puck is IN THE NET.
    
    # Normalize coords for goal check
    df_puck = df[df['entity_type'] == 'puck'].copy()
    if df_puck.empty:
        print("No puck data.")
        return

    # Assuming standard coordinates: Net at x=100 (or -100). 
    # Let's check bounds.
    x_max = df_puck['x'].abs().max()
    print(f"Puck X Max: {x_max}")
    
    # Check if we can infer time range relative to goal.
    # The 'goal' usually happens near end of clip.
    # Let's assume the last frame is roughly 'post-goal'.
    
    max_frame = df['frame_idx'].max()
    min_frame = df['frame_idx'].min()
    duration_frames = max_frame - min_frame
    duration_sec = duration_frames / 100.0 # Wait, Edge is usually ?
    # Edge is usually 30fps or similar? No, usually timestamp column exists.
    
    if 'timestamp' in df.columns:
        t_min = df['timestamp'].min()
        t_max = df['timestamp'].max()
        print(f"Timestamp Range: {t_min} to {t_max} (Diff: {t_max - t_min})")
        # Is timestamp system time or game time?
    
    if 'game_clock' in df.columns:
        gc_min = df['game_clock'].min()
        gc_max = df['game_clock'].max()
        print(f"Game Clock Range: {gc_min} to {gc_max}")
    
    # Heuristic: 
    # If blocks are ~14s before goal, and clip is 10s long, block is missing.
    
if __name__ == "__main__":
    debug_timing()
