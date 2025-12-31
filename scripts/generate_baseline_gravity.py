import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.edge import filter_data_to_goal_moment

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"

def generate_baseline():
    """Generate baseline by scanning ALL CSV files in season subdirectories."""
    all_points = []
    
    # Find all position CSV files in season subdirectories
    csv_files = []
    for season in ['20242025', '20252026']:
        season_dir = os.path.join(DATA_DIR, season)
        if os.path.exists(season_dir):
            csv_files.extend(glob.glob(os.path.join(season_dir, "*_positions.csv")))
    
    # Sort files to help with caching (files from same game together)
    csv_files.sort()
    
    # OUTPUT FILE
    output_file = 'data/intermediate_baseline_points.csv'
    if os.path.exists(output_file):
        print(f"Resuming from {output_file}...")
        df_existing = pd.read_csv(output_file)
        all_points = df_existing.values.tolist()
        # Find processed files (approximate logic, or just assume we add to it)
        # Better: keep a separate checklist file
    else:
        # Write header
        pd.DataFrame(columns=['x', 'y', 'mod']).to_csv(output_file, index=False)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    feed_cache = {}
    cache_lock = threading.Lock()
    file_lock = threading.Lock()
    
    print(f"Starting parallel processing of {len(csv_files)} files...")
    
    def process_file(pos_file):
        try:
            basename = os.path.basename(pos_file)
            parts = basename.replace('_positions.csv', '').split('_')
            if len(parts) < 4: return None
            
            game_id = parts[1]
            event_id = parts[3]
            
            # CACHED FEED ACCESS
            feed = None
            with cache_lock:
                 if game_id in feed_cache:
                     feed = feed_cache[game_id]
            
            if not feed:
                 from puck.nhl_api import get_game_feed
                 feed = get_game_feed(game_id)
                 with cache_lock:
                     # limit cache size
                     if len(feed_cache) > 10: feed_cache.clear() 
                     feed_cache[game_id] = feed
            
            # READ DATA
            df_pos = pd.read_csv(pos_file)
            if df_pos.empty: return None
            
            # Standardize units
            if df_pos['x'].abs().max() > 120:
                df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
                df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
            
            # NORMALIZE TO POSITIVE END
            last_frame = df_pos['frame_idx'].max()
            end_data = df_pos[(df_pos['entity_type'] == 'player') & (df_pos['frame_idx'] > last_frame - 50)]
            if end_data.empty: return None
            
            if end_data['x'].mean() < 0:
                df_pos['x'] = -df_pos['x']
                df_pos['y'] = -df_pos['y']

            df_pos = filter_data_to_goal_moment(df_pos)
            if df_pos.empty: return None
            
            # TEAM ID via API
            df_p = df_pos[df_pos['entity_type'] == 'player']
            if df_p.empty: return None
            
            plays = feed.get('plays', [])
            goal_evt = next((p for p in plays if str(p.get('eventId')) == event_id), None)
            
            off_team = None
            def_team = None
            
            if goal_evt:
                off_id_api = goal_evt.get('details', {}).get('eventOwnerTeamId')
                if off_id_api:
                    off_val = float(off_id_api)
                    teams = df_p['team_id'].unique()
                    
                    if off_val in teams: off_team = off_val
                    else:
                        match = next((t for t in teams if int(t) == int(off_val)), None)
                        if match: off_team = match
            
            if off_team is None: return None
            
            teams = df_p['team_id'].unique()
            def_team = next((t for t in teams if t != off_team), None)
            if def_team is None: return None
            
            off_p = df_p[df_p['team_id'] == off_team]
            def_p = df_p[df_p['team_id'] == def_team]
            
            # GOALIE EXCLUSION
            try:
                def_means = def_p.groupby('entity_id')['x'].mean()
                if not def_means.empty:
                    goalie_id = def_means.idxmax()
                    def_p = def_p[def_p['entity_id'] != goalie_id]
            except: pass
            
            if def_p.empty: return None
            
            # POSSESSION FILTER
            from puck.possession import infer_possession_events
            poss_events = infer_possession_events(df_pos, threshold_ft=6.0)
            poss_frames = set()
            if not poss_events.empty:
                off_ids = set(off_p['entity_id'].unique())
                for _, pev in poss_events.iterrows():
                     if (str(pev['player_id']) in off_ids) or (pev['player_id'] in off_ids):
                         for f in range(int(pev['start_frame']), int(pev['end_frame']) + 1):
                             poss_frames.add(f)
            
            if not poss_frames: return None
            def_frames = {f: g for f, g in def_p.groupby('frame_idx') if f in poss_frames}
            
            points = []
            for pid, p_track in off_p.groupby('entity_id'):
                for _, f_row in p_track.iterrows():
                    frame = f_row['frame_idx']
                    if frame not in def_frames: continue
                    
                    mx, my = f_row['x'], f_row['y']
                    df_frame = def_frames[frame]
                    dx = df_frame['x'] - mx
                    dy = df_frame['y'] - my
                    d = np.sqrt(dx**2 + dy**2).mean() # Mean distance to all defenders
                    
                    points.append([mx, my, d])
            return points

        except Exception as e:
            return None

    # EXECUTE PARALLEL
    batch_size = 100
    batch_points = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, f): f for f in csv_files}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  Processed {completed}/{len(csv_files)}...")
            
            res = future.result()
            if res:
                batch_points.extend(res)
            
            if len(batch_points) > 5000:
                with file_lock:
                    pd.DataFrame(batch_points).to_csv(output_file, mode='a', header=False, index=False)
                    all_points.extend(batch_points)
                    batch_points = []
    
    # Save remainders
    if batch_points:
        pd.DataFrame(batch_points).to_csv(output_file, mode='a', header=False, index=False)
        all_points.extend(batch_points)
                    

    print(f"Collected {len(all_points)} data points.")
    df_points = pd.DataFrame(all_points, columns=['x', 'y', 'mod'])
    
    # FILTER: Only offensive zone (X > 0) - goal sequences are oriented to attack positive X
    df_points = df_points[df_points['x'] > 0]
    print(f"After filtering to offensive zone (X > 0): {len(df_points)} points")
    
    # Grid Aggregation
    grid_size = 5
    df_points['x_bin'] = (df_points['x'] // grid_size) * grid_size
    df_points['y_bin'] = (df_points['y'] // grid_size) * grid_size
    
    baseline = df_points.groupby(['x_bin', 'y_bin'])['mod'].agg(['mean', 'count']).reset_index()
    baseline = baseline[baseline['count'] > 50]  # Minimum sample size
    
    baseline.to_csv(os.path.join(DATA_DIR, "mod_baseline.csv"), index=False)
    print(f"Saved baseline with {len(baseline)} cells.")
    
    # Simple heatmap viz
    plt.figure(figsize=(10, 8))
    pivot = baseline.pivot(index='y_bin', columns='x_bin', values='mean')
    plt.imshow(pivot, extent=[baseline['x_bin'].min(), baseline['x_bin'].max() + 5, 
                               baseline['y_bin'].min(), baseline['y_bin'].max() + 5], 
               origin='lower', cmap='RdYlGn_r')
    plt.colorbar(label='Mean Opponent Distance (ft)')
    plt.title(f"Baseline Defensive Density ({len(csv_files)} goals)")
    plt.xlabel("Rink X (ft)")
    plt.ylabel("Rink Y (ft)")
    plt.savefig(os.path.join(DATA_DIR, "baseline_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    generate_baseline()
