import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from puck.edge import transform_coordinates, filter_data_to_goal_moment
from puck.nhl_api import get_game_feed
from puck.possession import infer_possession_events
import matplotlib.pyplot as plt

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
OUTPUT_FILE = os.path.join(DATA_DIR, "gravity_analysis.csv")

def get_roster_map(game_id):
    try:
        feed = get_game_feed(game_id)
        if not feed: return {}
        
        # Team ID to Abbr map
        teamm_map = {}
        for side in ['homeTeam', 'awayTeam']:
            tdata = feed.get(side, {})
            tid = tdata.get('id')
            tabbr = tdata.get('abbrev')
            if tid and tabbr:
                teamm_map[int(tid)] = tabbr

        roster = {}
        # 1. playerByTeam (Most structured in new API)
        if 'playerByTeam' in feed:
            for side in ['homeTeam', 'awayTeam']:
                pbt = feed['playerByTeam'].get(side, {})
                # Find team abbr for this side
                tabbr = teamm_map.get(int(feed.get(side, {}).get('id', 0))) if feed.get(side) else None

                for key in ['roster', 'rosterSpots', 'players']:
                    if key in pbt:
                        for p in pbt[key]:
                            pid = p.get('player_id') or p.get('id') or p.get('playerId')
                            if not pid: continue
                            
                            # Name / Position resolution (simplified for brevity)
                            name = p.get('fullName') or f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip()
                            pos = p.get('positionCode') or 'UNK'
                            
                            roster[str(pid)] = {'name': name, 'position': pos, 'team': tabbr}
                            try: roster[int(pid)] = roster[str(pid)]
                            except: pass
                            
        # 2. rosterSpots (Flat list)
        if 'rosterSpots' in feed:
            for p in feed['rosterSpots']:
                pid = p.get('playerId') or p.get('id')
                if not pid: continue
                if str(pid) in roster: continue # Already found in playerByTeam
                
                name = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip()
                pos = p.get('positionCode') or 'UNK'
                # Find team? Sometimes rosterSpots has teamId
                tid = p.get('teamId')
                tabbr = teamm_map.get(int(tid)) if tid else None
                
                roster[str(pid)] = {'name': name, 'position': pos, 'team': tabbr}
                try: roster[int(pid)] = roster[str(pid)]
                except: pass

        return roster
    except Exception as e:
        return {}

def visualize_gravity(df_agg, output_dir):
    try:
        # Ensure numeric columns
        cols = [
            'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft',
            'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
            'goals_on_ice_count'
        ]
        for col in cols:
            df_agg[col] = pd.to_numeric(df_agg[col], errors='coerce')
        
        # Filter Goalies
        df_agg = df_agg[df_agg['position'] != 'G']
        
        seasons = df_agg['season'].unique()
        print(f"\nGenerating plots for seasons: {seasons}")
        
        groups = {
            'Forwards': ['C', 'L', 'R', 'LW', 'RW'],
            'Defense': ['D', 'LD', 'RD']
        }
        
        for season in seasons:
            df_season = df_agg[df_agg['season'] == season]
            if df_season.empty: continue
            
            for group_name, positions in groups.items():
                df_group = df_season[df_season['position'].isin(positions)].copy()
                if df_group.empty: continue

                # 1. On-Puck Plot
                _create_scatter(
                    df_group, 
                    'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft',
                    f"On-Puck Gravity ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_plot_{season}_{group_name.lower()}_on_puck.png"),
                    'coral' if group_name == 'Forwards' else 'cornflowerblue'
                )

                # 2. Off-Puck Plot
                _create_scatter(
                    df_group, 
                    'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
                    f"Off-Puck (Threat) Gravity ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_plot_{season}_{group_name.lower()}_off_puck.png"),
                    'orange' if group_name == 'Forwards' else 'royalblue'
                )

                # 3. Correlation Plot
                _create_correlation(
                    df_group,
                    'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft',
                    f"Relative On-Puck vs Off-Puck MOD ({season}) - {group_name}",
                    os.path.join(output_dir, f"gravity_rel_corr_{season}_{group_name.lower()}.png")
                )
            
    except Exception as e:
        print(f"Visualization error: {e}")

def _create_scatter(df, x_col, y_col, title, out_path, color):
    # Filter valid rows
    df_valid = df.dropna(subset=[x_col, y_col])
    if df_valid.empty: return

    plt.figure(figsize=(12, 8))
    sizes = df_valid['goals_on_ice_count'] * 30
    sizes = sizes.clip(lower=20, upper=300)
    
    plt.scatter(
        df_valid[x_col], df_valid[y_col], 
        s=sizes, alpha=0.6, edgecolors='w', linewidth=0.5, color=color
    )
    
    for _, row in df_valid.iterrows():
        lbl = str(row['player_name'])
        if len(lbl) > 15: lbl = lbl.split(' ')[-1]
        plt.annotate(
            lbl, (row[x_col], row[y_col]),
            fontsize=9, alpha=0.9, xytext=(3, 3), textcoords='offset points'
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Relative Mean Opponent Distance (ft)\n<-- Higher Relative Pressure | More Relative Space -->", fontsize=12)
    plt.ylabel("Relative Avg Nearest Opponent Distance (ft)\n<-- In Traffic | Finding Open Space -->", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add zero lines
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def _create_correlation(df, x_col, y_col, title, out_path):
    df_valid = df.dropna(subset=[x_col, y_col])
    if df_valid.empty: return

    plt.figure(figsize=(10, 10))
    plt.scatter(df_valid[x_col], df_valid[y_col], alpha=0.7, color='purple')
    
    # Add diagonal line
    mi = min(df_valid[x_col].min(), df_valid[y_col].min())
    ma = max(df_valid[x_col].max(), df_valid[y_col].max())
    plt.plot([mi, ma], [mi, ma], 'k--', alpha=0.2)
    
    for _, row in df_valid.iterrows():
        lbl = row['player_name'].split(' ')[-1]
        plt.annotate(lbl, (row[x_col], row[y_col]), fontsize=8)
        
    plt.title(title, fontsize=14)
    plt.xlabel("Relative On-Puck MOD (ft)", fontsize=11)
    plt.ylabel("Relative Off-Puck MOD (ft)", fontsize=11)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Add zero lines
    plt.axvline(0, color='black', alpha=0.3)
    plt.axhline(0, color='black', alpha=0.3)
    
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def analyze_gravity():
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # 1. SCAN FILES
    csv_files = []
    for season in ['20242025', '20252026']:
        season_dir = os.path.join(DATA_DIR, season)
        if os.path.exists(season_dir):
            for f in glob.glob(os.path.join(season_dir, "*_positions.csv")):
                basename = os.path.basename(f)
                parts = basename.replace('_positions.csv', '').split('_')
                if len(parts) >= 4 and parts[0] == 'game' and parts[2] == 'goal':
                    game_id = parts[1]
                    if len(game_id) >= 6 and game_id[4:6] == '01': continue # Skip pre-season
                    csv_files.append({'path': f, 'game_id': game_id, 'event_id': parts[3], 'season': season})
    
    print(f"Found {len(csv_files)} goal CSV files to analyze.")

    # 2. SETUP RESOURCES (DUAL BASELINES)
    try:
        df_base_on = pd.read_csv(os.path.join(DATA_DIR, "mod_baseline_on_puck.csv"))
        base_on_map = df_base_on.set_index(['x_bin', 'y_bin'])['mean'].to_dict()
        
        df_base_off = pd.read_csv(os.path.join(DATA_DIR, "mod_baseline_off_puck.csv"))
        base_off_map = df_base_off.set_index(['x_bin', 'y_bin'])['mean'].to_dict()
        print("Loaded dual baselines (On-Puck & Off-Puck).")
    except Exception as e:
        print(f"Error loading baselines (Wait for generation to finish!): {e}")
        return

    feed_cache = {}
    roster_cache = {}
    cache_lock = threading.Lock()
    file_lock = threading.Lock()
    
    # CHECKPOINTING
    processed_keys = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            if not df_existing.empty:
                for _, row in df_existing.iterrows():
                    processed_keys.add((str(row['game_id']), str(row['event_id'])))
            print(f"Resuming analysis. Already processed {len(processed_keys)} records.")
        except:
            print("Creating new output file.")
            pd.DataFrame(columns=['season', 'game_id', 'event_id', 'player_id', 'player_name', 'team_abbr', 'position', 'game_state',
                                   'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft', 'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
                                   'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft', 'on_puck_frames', 'off_puck_frames']).to_csv(OUTPUT_FILE, index=False)
    else:
        pd.DataFrame(columns=['season', 'game_id', 'event_id', 'player_id', 'player_name', 'team_abbr', 'position', 'game_state',
                               'on_puck_mean_dist_ft', 'on_puck_nearest_dist_ft', 'off_puck_mean_dist_ft', 'off_puck_nearest_dist_ft',
                               'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft', 'on_puck_frames', 'off_puck_frames']).to_csv(OUTPUT_FILE, index=False)

    # 3. DEFINE PROCESSING
    def process_goal_file(file_info):
        game_id = file_info['game_id']
        event_id = file_info['event_id']
        season = file_info['season']
        pos_file = file_info['path']

        # Skip if already processed
        if (str(game_id), str(event_id)) in processed_keys:
            return []

        try:
            # Cache feed/roster
            with cache_lock:
                feed = feed_cache.get(game_id)
                roster = roster_cache.get(game_id)
            
            if not feed:
                feed = get_game_feed(game_id)
                roster = get_roster_map(game_id)
                with cache_lock:
                    if len(feed_cache) > 50:
                        feed_cache.clear()
                        roster_cache.clear()
                    feed_cache[game_id] = feed
                    roster_cache[game_id] = roster

            df_pos = pd.read_csv(pos_file)
            if df_pos.empty: return []

            # Units
            if df_pos['x'].abs().max() > 120:
                 df_pos['x'] = (df_pos['x'] - 1200.0) / 12.0
                 df_pos['y'] = -(df_pos['y'] - 510.0) / 12.0
            
            df_pos['entity_id'] = df_pos['entity_id'].astype(str)
            
            # Attacking End Normalization
            last_frame = df_pos['frame_idx'].max()
            end_data = df_pos[(df_pos['entity_type'] == 'player') & (df_pos['frame_idx'] > last_frame - 50)]
            if not end_data.empty and end_data['x'].mean() < 0:
                df_pos['x'] = -df_pos['x']
                df_pos['y'] = -df_pos['y']

            df_pos = filter_data_to_goal_moment(df_pos)
            df_players = df_pos[df_pos['entity_type'] == 'player']
            if df_players.empty: return []
            
            # Team / Game State
            plays = feed.get('plays', [])
            play = next((p for p in plays if str(p.get('eventId')) == event_id), None)
            if not play: return []
            
            off_team_id = play.get('details', {}).get('eventOwnerTeamId')
            if not off_team_id: return []
            
            scoring_team_id = float(off_team_id)
            sc = play.get('situationCode', '')
            game_state = '5v5' if (len(sc) == 4 and sc[1] == '5' and sc[2] == '5') else 'OTHER'
            if game_state != '5v5': return []

            # Filter Defenders (Exclude Goalie)
            off_pids = df_players[df_players['team_id'] == scoring_team_id]['entity_id'].unique()
            def_df = df_players[df_players['team_id'] != scoring_team_id]
            if def_df.empty: return []

            def_pids = def_df['entity_id'].unique()
            goalie_ids = [pid for pid in def_pids if (roster.get(str(pid)) or {}).get('position') == 'G']
            if goalie_ids:
                def_df = def_df[~def_df['entity_id'].isin(goalie_ids)]
                if def_df.empty: return []

            # Possession / Distance calculation
            poss_events = infer_possession_events(df_pos, threshold_ft=6.0)
            poss_map = {}
            if not poss_events.empty:
                off_pids_set = set(str(p) for p in off_pids)
                for _, pev in poss_events.iterrows():
                    # Only count possession if the player is on the OFFENSIVE team
                    pid_str = str(pev['player_id'])
                    if pev['is_possession'] and pid_str in off_pids_set:
                        for f in range(int(pev['start_frame']), int(pev['end_frame']) + 1):
                            poss_map[f] = pid_str

            results = []
            for pid in off_pids:
                p_track = df_players[df_players['entity_id'] == pid].set_index('frame_idx')
                common_frames = p_track.index.intersection(def_df['frame_idx'].unique())
                
                points = {'on_puck': [], 'off_puck': []}
                for frame in common_frames:
                    poss_pid = poss_map.get(frame)
                    if not poss_pid: continue
                    label = 'on_puck' if poss_pid == pid else 'off_puck'
                    
                    try:
                        me = p_track.loc[frame]
                        if isinstance(me, pd.DataFrame): me = me.iloc[0]
                        mx, my = me['x'], me['y']
                        
                        defs = def_df[def_df['frame_idx'] == frame]
                        dists = np.sqrt((defs['x'] - mx)**2 + (defs['y'] - my)**2)
                        
                        # Baseline lookup
                        xb, yb = (mx // 5) * 5, (my // 5) * 5
                        
                        # DUAL BASELINE LOGIC
                        if label == 'on_puck':
                             exp = base_on_map.get((xb, yb), np.nan)
                        else:
                             exp = base_off_map.get((xb, yb), np.nan)

                        points[label].append((dists.mean(), dists.min(), exp))
                    except: pass
                
                if not points['on_puck'] and not points['off_puck']: continue

                def _avg(lst, idx):
                    vals = [x[idx] for x in lst if not np.isnan(x[idx])]
                    return np.mean(vals) if vals else np.nan

                mod_on, mod_off = _avg(points['on_puck'], 0), _avg(points['off_puck'], 0)
                exp_on, exp_off = _avg(points['on_puck'], 2), _avg(points['off_puck'], 2)

                results.append({
                    'season': season, 'game_id': game_id, 'event_id': event_id,
                    'player_id': pid, 'player_name': (roster.get(str(pid)) or {}).get('name', f"Player {pid}"),
                    'team_abbr': (roster.get(str(pid)) or {}).get('team', 'UNK'),
                    'position': (roster.get(str(pid)) or {}).get('position', 'UNK'),
                    'game_state': game_state,
                    'on_puck_mean_dist_ft': mod_on,
                    'on_puck_nearest_dist_ft': _avg(points['on_puck'], 1),
                    'off_puck_mean_dist_ft': mod_off,
                    'off_puck_nearest_dist_ft': _avg(points['off_puck'], 1),
                    'rel_on_puck_mean_dist_ft': (mod_on - exp_on) if not np.isnan(mod_on + exp_on) else np.nan,
                    'rel_off_puck_mean_dist_ft': (mod_off - exp_off) if not np.isnan(mod_off + exp_off) else np.nan,
                    'on_puck_frames': len(points['on_puck']),
                    'off_puck_frames': len(points['off_puck'])
                })
            return results
        except Exception as e:
            return []

    # 4. EXECUTE
    print(f"Executing parallel analysis...")
    batch = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_goal_file, f): f for f in csv_files}
        for i, f in enumerate(as_completed(futures)):
            if i % 100 == 0: print(f"  Processed {i}/{len(csv_files)}...")
            res = f.result()
            if res: batch.extend(res)
            
            if len(batch) > 300:
                with file_lock:
                    pd.DataFrame(batch).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    batch = []

    if batch:
        with file_lock:
            pd.DataFrame(batch).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    # 5. AGGREGATE
    print("\nAggregating seasonal results...")
    df_results = pd.read_csv(OUTPUT_FILE)
    if df_results.empty: return

    agg_dir = os.path.join("analysis", "gravity")
    os.makedirs(agg_dir, exist_ok=True)
    
    df_agg = df_results.groupby(['season', 'player_id']).agg({
        'player_name': 'first',
        'team_abbr': 'first',
        'position': 'first',
        'on_puck_mean_dist_ft': 'mean',
        'on_puck_nearest_dist_ft': 'mean',
        'off_puck_mean_dist_ft': 'mean',
        'off_puck_nearest_dist_ft': 'mean',
        'rel_on_puck_mean_dist_ft': 'mean',
        'rel_off_puck_mean_dist_ft': 'mean',
        'game_id': 'count'
    }).rename(columns={'game_id': 'goals_on_ice_count'}).reset_index()
    
    df_agg.to_csv(os.path.join(agg_dir, "player_gravity_season.csv"), index=False)
    visualize_gravity(df_agg, agg_dir)

if __name__ == "__main__":
    analyze_gravity()
