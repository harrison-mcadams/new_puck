import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import random

# Setup paths
sys.path.append(os.getcwd())
try:
    from puck import rink, nhl_api
except ImportError:
    print("Warning: puck modules not found")

# Helper to get clock time
def get_clock_string(period, time_in_period):
    p_str = "OT" if period > 3 else f"P{period}"
    return f"{p_str} {time_in_period}"

def calculate_kinematics(df):
    """Calculate velocity, speed, and angle for puck data."""
    # Ensure sorted
    df = df.sort_values('frame_idx')
    
    # Calculate deltas
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['timestamp'].diff() / 10.0 # Deciseconds -> Seconds
    
    # Fill gaps (simple method for now)
    df['dt'] = df['dt'].replace(0, np.nan).fillna(0.1)
    
    # Velocity
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    
    # Speed
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Angle (radians)
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    
    # Angle Change (delta angle)
    # We need to handle wrap-around for angle diffs
    # Use complex numbers for robust angle difference
    # or just simple diff and wrap to -pi, pi
    a1 = df['angle']
    a2 = df['angle'].shift(1)
    diff = a1 - a2
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    df['d_angle'] = diff
    
    return df

def segment_shot_vectors(df, angle_threshold_deg=15.0, min_speed=10.0, min_len=5):
    """
    Segment the trajectory into ballistic 'vectors'.
    """
    segments = []
    
    if df.empty:
        return segments
        
    current_segment = [df.iloc[0]]
    angle_thresh_rad = np.radians(angle_threshold_deg)
    
    for i in range(1, len(df)):
        frame = df.iloc[i]
        prev = df.iloc[i-1]
        
        angle_diff = abs(frame['d_angle'])
        speed_ok = frame['speed'] > min_speed
        
        speed_diff = frame['speed'] - prev['speed']
        new_impulse = speed_diff > 5.0
        
        if angle_diff < angle_thresh_rad and speed_ok and not new_impulse:
            current_segment.append(frame)
        else:
            if len(current_segment) >= min_len:
                segments.append(pd.DataFrame(current_segment))
            current_segment = [frame]
            
    if len(current_segment) >= min_len:
        segments.append(pd.DataFrame(current_segment))
        
    return segments

def analyze_random_batch(n=5):
    """Analyze a batch of random goal files to test transferability."""
    
    # Locate all edge goal positions files
    search_path = os.path.join('data', 'edge_goals', '20242025', '*_positions.csv')
    all_files = glob.glob(search_path)
    
    if not all_files:
        print(f"No files found in {search_path}")
        return
        
    print(f"Found {len(all_files)} total goal clips. Selecting {n} random samples.")
    
    selected_files = random.sample(all_files, min(n, len(all_files)))
    feed_cache = {}

    for idx, pos_csv in enumerate(selected_files):
        # Extract Game/Event ID from filename
        basename = os.path.basename(pos_csv)
        parts = basename.replace('_positions.csv', '').split('_')
        
        if len(parts) >= 4:
            game_id = parts[1]
            event_id = parts[3]
        else:
            game_id = "Unknown"
            event_id = "Unknown"
            
        # --- Metadata Fetching ---
        meta_str = f"Game {game_id} Event {event_id}"
        goal_title = meta_str
        
        if game_id != "Unknown":
            if game_id not in feed_cache:
                try:
                    feed_cache[game_id] = nhl_api.get_game_feed(int(game_id))
                except:
                    feed_cache[game_id] = {}
        
            feed = feed_cache.get(game_id, {})
            plays = feed.get('plays', [])
            goal_play = next((p for p in plays if str(p.get('eventId')) == str(event_id)), None)
            
            if goal_play:
                period = goal_play.get('periodDescriptor', {}).get('number', 1)
                time_in_period = goal_play.get('timeInPeriod', '00:00')
                clock = get_clock_string(period, time_in_period)
                scorer_id = goal_play.get('details', {}).get('scoringPlayerId')
                game_date = feed.get('gameDate', 'Unknown Date')
                
                goal_title = f"{game_date} | {clock}\nGame {game_id} | Goal {event_id} | ScorerID: {scorer_id}"
                
        print(f"\n--- Processing {idx+1}/{n}: {goal_title} ---")
        
        df_pos = pd.read_csv(pos_csv)
        df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()
        
        if df_puck.empty:
            print("  No puck data found.")
            continue
            
        # 1. Kinematics
        df_puck = calculate_kinematics(df_puck)
        
        # 2. Segment
        segments = segment_shot_vectors(df_puck)
        
        print(f"  Segments Found: {len(segments)}")
        
        # 3. Visualize
        fig, ax = plt.subplots(figsize=(14, 8))
        try:
            rink.draw_rink(ax)
        except:
            pass
            
        colors = plt.cm.jet(np.linspace(0, 1, len(segments)))
        
        stats = []
        for si, seg in enumerate(segments):
            stats.append({
                'id': si,
                'speed': seg['speed'].mean(),
                'len': len(seg),
                'angle': np.degrees(seg['angle'].mean())
            })
            
        shot_candidate = max(stats, key=lambda x: x['speed'] if x['len'] > 2 else 0) if stats else None
        shot_idx = shot_candidate['id'] if shot_candidate else -1
        
        table_data = []
        
        for i, seg in enumerate(segments):
            label = f"Vec {i}"
            if i == shot_idx: 
                label += " (SHOT)"
            elif i == shot_idx + 1:
                label += " (DEFLECT?)"
                
            ax.plot(seg['x'], seg['y'], '.-', color=colors[i], label=label)
            ax.plot(seg.iloc[0]['x'], seg.iloc[0]['y'], 'o', color=colors[i], markersize=8)
            mid_idx = len(seg)//2
            mid = seg.iloc[mid_idx]
            ax.text(mid['x'], mid['y']+2, f"{i}", color=colors[i], fontsize=12, fontweight='bold')
            
            table_data.append([i, f"{stats[i]['speed']:.1f}", f"{stats[i]['len']}", f"{stats[i]['angle']:.1f}", label])

        ax.set_title(f"Shot Vector Segmentation\n{goal_title}")
        ax.legend()
        
        if table_data:
            plt.table(cellText=table_data, colLabels=["Vec", "Avg Speed", "Frames", "Angle", "Type"], 
                      loc='bottom', cellLoc='center', bbox=[0.0, -0.25, 1.0, 0.15])
        plt.subplots_adjust(bottom=0.25)
        
        out_name = f"shot_vector_{game_id}_{event_id}.png"
        out_path = os.path.join('analysis', 'plots', out_name)
        os.makedirs(os.path.join('analysis', 'plots'), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
        print(f"  Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    analyze_random_batch(5)
