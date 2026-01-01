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
    
    # OUTPUT FILES
    out_on = 'data/intermediate_on_puck.csv'
    out_off = 'data/intermediate_off_puck.csv'
    
    # Initialize Headers if not exist
    for f in [out_on, out_off]:
        if not os.path.exists(f):
             pd.DataFrame(columns=['x', 'y', 'mod']).to_csv(f, index=False)

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
            
            # Map Frame -> PlayerID who has puck
            frame_possessor = {}
            if not poss_events.empty:
                off_ids = set(off_p['entity_id'].unique())
                for _, pev in poss_events.iterrows():
                     pid = pev['player_id']
                     # Ensure possessor is on offensive team (sometimes defense touches it)
                     # Actually, if defense has it, it's not offensive possession.
                     if (str(pid) in off_ids) or (pid in off_ids):
                         for f in range(int(pev['start_frame']), int(pev['end_frame']) + 1):
                             frame_possessor[f] = float(pid)
            
            if not frame_possessor: return None
            
            def_frames = {f: g for f, g in def_p.groupby('frame_idx') if f in frame_possessor}
            
            pts_on = []
            pts_off = []
            
            for pid, p_track in off_p.groupby('entity_id'):
                pid_val = float(pid)
                for _, f_row in p_track.iterrows():
                    frame = f_row['frame_idx']
                    if frame not in def_frames: continue
                    
                    possessor = frame_possessor[frame] # Guaranteed to exist due to check
                    
                    mx, my = f_row['x'], f_row['y']
                    df_frame = def_frames[frame]
                    dx = df_frame['x'] - mx
                    dy = df_frame['y'] - my
                    d = np.sqrt(dx**2 + dy**2).mean() 
                    
                    if pid_val == possessor:
                        pts_on.append([mx, my, d])
                    else:
                        pts_off.append([mx, my, d])
                        
            return {'on': pts_on, 'off': pts_off}

        except Exception as e:
            return None

    # EXECUTE PARALLEL
    batch_on = []
    batch_off = []
    
    count_on = 0
    count_off = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, f): f for f in csv_files}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  Processed {completed}/{len(csv_files)}...")
            
            res = future.result()
            if res:
                batch_on.extend(res['on'])
                batch_off.extend(res['off'])
            
            if len(batch_on) > 5000 or len(batch_off) > 5000:
                with file_lock:
                    if batch_on:
                        pd.DataFrame(batch_on).to_csv(out_on, mode='a', header=False, index=False)
                        count_on += len(batch_on)
                        batch_on = []
                    if batch_off:
                        pd.DataFrame(batch_off).to_csv(out_off, mode='a', header=False, index=False)
                        count_off += len(batch_off)
                        batch_off = []
    
    # Save remainders
    if batch_on:
        pd.DataFrame(batch_on).to_csv(out_on, mode='a', header=False, index=False)
        count_on += len(batch_on)
    if batch_off:
        pd.DataFrame(batch_off).to_csv(out_off, mode='a', header=False, index=False)
        count_off += len(batch_off)
                    

    print(f"Collection Complete. On-Puck: {count_on}, Off-Puck: {count_off}")
    
    # GENERATE BASELINES
    def make_baseline(input_csv, output_base, title):
        print(f"Generating baseline for {title}...")
        try:
            df = pd.read_csv(input_csv)
            df = df[df['x'] > 0] # Offensive zone only
            
            grid_size = 5
            df['x_bin'] = (df['x'] // grid_size) * grid_size
            df['y_bin'] = (df['y'] // grid_size) * grid_size
            
            baseline = df.groupby(['x_bin', 'y_bin'])['mod'].agg(['mean', 'count']).reset_index()
            baseline = baseline[baseline['count'] > 20] 
            
            baseline.to_csv(output_base, index=False)
            
            # Viz
            plt.figure(figsize=(10, 8))
            pivot = baseline.pivot(index='y_bin', columns='x_bin', values='mean')
            plt.imshow(pivot, extent=[baseline['x_bin'].min(), baseline['x_bin'].max() + 5, 
                                       baseline['y_bin'].min(), baseline['y_bin'].max() + 5], 
                       origin='lower', cmap='RdYlGn_r')
            plt.colorbar(label='Mean Opponent Distance (ft)')
            plt.title(f"Baseline: {title}")
            plt.xlabel("Rink X")
            plt.ylabel("Rink Y")
            plt.savefig(output_base.replace('.csv', '.png'))
            plt.close()
            print(f"Saved {output_base}")
        except Exception as e:
            print(f"Failed to generate {title}: {e}")

    make_baseline(out_on, os.path.join(DATA_DIR, "mod_baseline_on_puck.csv"), "On-Puck Gravity")
    make_baseline(out_off, os.path.join(DATA_DIR, "mod_baseline_off_puck.csv"), "Off-Puck Gravity")

if __name__ == "__main__":
    generate_baseline()
