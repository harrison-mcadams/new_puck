import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config
from puck import analyze
from puck import timing
from puck.rink import draw_rink
from puck.plot import add_summary_text, plot_events
from puck.analyze import compute_relative_map, league

def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='20252026')
    args = parser.parse_args()
    season = args.season
    
    out_dir_base = 'analysis/players'
    league_out_dir = os.path.join(out_dir_base, f'{season}/league')
    os.makedirs(league_out_dir, exist_ok=True)
    
    print(f"--- Cached Player Analysis for {season} ---")
    print(f"DEBUG: Process ID: {os.getpid()}")
    
    # 1. Load DataFrame (needed for scatter plots of events)
    print("Loading season data...")
    df_data = timing.load_season_df(season)
    if df_data is None or df_data.empty:
        print("No data found.")
        return

    # Helper maps
    # We construct them from df_data to ensure coverage
    cols = ['player_id', 'player_name', 'home_abb', 'away_abb', 'home_id', 'away_id']
    if 'player_number' in df_data.columns: cols.append('player_number')
    
    # Player Name/Number Map
    # Group by ID and take first non-null
    p_info = df_data[['player_id', 'player_name']].dropna().drop_duplicates('player_id').set_index('player_id')
    pid_name_map = p_info['player_name'].to_dict()
    
    # Team Map (ID -> Abb)
    t1 = df_data[['home_id', 'home_abb']].rename(columns={'home_id': 'id', 'home_abb': 'abb'})
    t2 = df_data[['away_id', 'away_abb']].rename(columns={'away_id': 'id', 'away_abb': 'abb'})
    t_map = pd.concat([t1, t2]).drop_duplicates('id').set_index('id')['abb'].to_dict()

    # Drop heavy columns to save RAM
    drop_cols = ['event_description', 'game_date']
    for c in drop_cols:
        if c in df_data.columns:
            # removing event_description is fine, but keep game_date if needed for verify? no.
            df_data.drop(columns=[c], inplace=True)
    # df_data now held in memory for raw plotting
    gc.collect()

    # 2. Identify Players & Updates
    # We only care about 5v5 for now to match old script
    # 2. Identify Players & Updates
    # We only care about 5v5 for now to match old script
    COND = '5v5'
    base_cond = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # Filter df_data globally for this condition to ensure raw plots are 5v5 only
    from puck.parse import build_mask
    if not df_data.empty:
        mask = build_mask(df_data, base_cond)
        df_data = df_data[mask].copy()
        print(f"DEBUG: Filtered df_data size: {len(df_data)}")
        if len(df_data) > 0:
            print(f"DEBUG: Sample game_ids in global df: {df_data['game_id'].unique()[:5]}")
        gc.collect()
    
    # Load Manifest
    manifest_path = os.path.join(league_out_dir, 'map_manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except: pass
        
    # Get current game counts
    player_games = df_data.groupby('player_id')['game_id'].agg(['count', 'max'])
    player_games.columns = ['games_played', 'last_game_id']
    
    # Filter players needing update
    pids_to_process = []
    min_games = 5
    
    for pid, row in player_games.iterrows():
        # Check Manifest
        cached = manifest.get(str(pid))
        
        # Check files existence (assuming we know team? - we'll check output later)
        # We don't verify file existence here perfectly without team loop, checking manifest is faster.
        # If forced refresh needed, user can delete manifest.
        
        needs_update = True
        if cached:
            if cached.get('games_played') == row['games_played'] and \
               cached.get('last_game_id') == int(row['last_game_id']):
                needs_update = False
                
        if needs_update and row['games_played'] >= 1: # Process even with 1 game, plot filtered by min_games later
            pids_to_process.append(pid)
            
    print(f"Found {len(pids_to_process)} players needing updates.")
    
    # Sanitize PIDs to ensure uniqueness and type consistency
    pids_to_process = sorted(list(set([int(p) for p in pids_to_process if str(p).replace('.0','').isdigit()])))
    # pids_to_process = [8483085]
    # print(f"DEBUG: FORCED SINGLE PID: {pids_to_process}")
    print(f"Sanitized PIDs: {len(pids_to_process)} unique integers.")
    print(f"DEBUG_UNIQUENESS: Total={len(pids_to_process)}, Unique={len(set(pids_to_process))}")
    if len(pids_to_process) != len(set(pids_to_process)):
         from collections import Counter
         c = Counter(pids_to_process)
         print(f"DEBUG: Duplicates found: {c.most_common(5)}")
    
    print(f"DEBUG: df_data['player_id'] type: {df_data['player_id'].dtype}")

    # 3. Load League Baseline (for relative maps)
    # Using analyze.league from cache (or standard)
    # If run_league_stats runs BEFORE this, baseline should be on disk.
    print("Loading league baseline...")
    baseline_path = os.path.join('analysis', 'league', season, COND)
    # We can try to load directly from .npy if analyze.league saved it
    try:
        baseline_res = league(season=season, mode='load', condition=base_cond, baseline_path=baseline_path)
        league_map = baseline_res.get('combined_norm')
        league_map_right = baseline_res.get('combined_norm_right')
    except Exception as e:
        print(f"Warning: Failed to load league baseline (maybe run league stats first?): {e}")
        league_map = None
        league_map_right = None

    # 4. Processing Loop (Chunks)
    chunk_size = config.BATCH_SIZE
    pids_sorted = sorted(pids_to_process)
    print(f"DEBUG: Final pids_sorted len={len(pids_sorted)}")
    print(f"DEBUG: Final pids_sorted unique len={len(set(pids_sorted))}")
    print(f"DEBUG: Start of pids_sorted: {pids_sorted[:10]}")
    
    # Resolve Partial Files
    cache_dir = os.path.join(config.get_cache_dir(season), 'partials')
    print(f"Checking cache dir: {cache_dir}")
    all_partials = sorted([f for f in os.listdir(cache_dir) if f.endswith(f'_{COND}.npz')]) if os.path.exists(cache_dir) else []
    
    if not all_partials:
        print(f"Warning: No partial files found for {COND} in {cache_dir}")
    
    # Map GameID -> Path
    # filename format: {game_id}_{cond}.npz
    game_path_map = {}
    for f in all_partials:
        gid = f.split('_')[0]
        try:
            game_path_map[int(gid)] = os.path.join(cache_dir, f)
        except: pass
        
    stats_accumulator = []

    # Processing Loop
    processed_pids = set()
    for i in range(0, len(pids_sorted), chunk_size):
        chunk = pids_sorted[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}: {len(chunk)} players...")
        print(f"DEBUG: Chunk values: {chunk}")
        
        # Aggregate Data for Chunk
        # Dict: pid -> { 'grid_team': ..., 'grid_other': ..., 'stats': {} }
        agg_data = {pid: {'grid_team': None, 'grid_other': None, 'stats': []} for pid in chunk}
        
        # Identify relevant games for this chunk to minimize I/O?
        # A set of game_ids where ANY player in chunk played.
        chunk_games = set()
        for pid in chunk:
            # We can use df_data to find games
            pgs = df_data[df_data['player_id'] == pid]['game_id'].unique()
            chunk_games.update(pgs)
            
        # Scan games
        loaded_games = 0
        for gid in chunk_games:
            path = game_path_map.get(gid)
            if not path: continue
            
            try:
                # Load NPZ
                with np.load(path) as data:
                    if 'empty' in data: continue
                    
                    # For each player in chunk
                    for pid in chunk:
                        # Ensure PID is int for key lookup
                        try:
                            pid_int = int(pid)
                        except:
                            pid_int = pid
                            
                        # Check keys
                        k_grid = f"p_{pid_int}_grid_team" 
                        # or grid_other
                        # Wait, process_daily_cache saves 'p_{pid}_grid_team' and 'p_{pid}_grid_other'
                        
                        if k_grid not in data: 
                             # print(f"DEBUG: {k_grid} not in {path}")
                             continue
                        
                        # print(f"DEBUG: Found {k_grid}")
                        
                        # Sum Grids
                        g_tm = data[f"p_{pid_int}_grid_team"]
                        g_ot = data[f"p_{pid_int}_grid_other"] if f"p_{pid_int}_grid_other" in data else None
                        
                        if agg_data[pid]['grid_team'] is None:
                            agg_data[pid]['grid_team'] = g_tm.astype(np.float64)
                        else:
                            agg_data[pid]['grid_team'] += g_tm
                            
                        if g_ot is not None:
                            if agg_data[pid]['grid_other'] is None:
                                agg_data[pid]['grid_other'] = g_ot.astype(np.float64)
                            else:
                                agg_data[pid]['grid_other'] += g_ot
                                
                        # Stats
                        k_stats = f"p_{pid_int}_stats"
                        if k_stats in data:
                            s_str = str(data[k_stats])
                            # It was saved as json string in npz?
                            # np.load of string usually returns a 0-d array.
                            s_dict = json.loads(s_str)
                            agg_data[pid]['stats'].append(s_dict)
                            
                loaded_games += 1
            except Exception:
                pass
                
        # Generate Plots for Chunk
        for pid in chunk:
            # FIX: Update pid_int for this loop iteration
            try:
                pid_int = int(pid)
            except:
                pid_int = pid
                
            data = agg_data[pid]
            if not data['stats']: 
                print(f"DEBUG: No stats for {pid}")
                continue
            
            # Aggregate stats
             # Sum numerical fields
            total_stats = {}
            keys_to_sum = ['team_xgs', 'other_xgs', 'team_goals', 'other_goals', 'team_attempts', 'other_attempts', 'team_seconds']
            
            for s in data['stats']:
                for k in keys_to_sum:
                    total_stats[k] = total_stats.get(k, 0.0) + s.get(k, 0.0)
            
            # Derived
            seconds = total_stats.get('team_seconds', 0.0)
            if seconds <= 0: continue
            
            xg_for = total_stats.get('team_xgs', 0.0)
            xg_ag = total_stats.get('other_xgs', 0.0)
            xg_for_60 = (xg_for / seconds) * 3600
            xg_ag_60 = (xg_ag / seconds) * 3600
            
            # Determine Team
            pname = pid_name_map.get(pid_int, f"Player {pid_int}") # Use pid_int here too
            
            # Defense against phantom duplication
            if pid in processed_pids:
                print(f"DEBUG: SKIPPING {pid} (Already Processed)")
                continue
            processed_pids.add(pid)
            
            # Stats dict missing 'team' key apparently.
            # Derive from df_data events
            # We already filter p_df later, but we need team now for directory.
            # Let's peek at df_data for this pid
            # We can use p_df logic here or just do a quick lookup
            # But we haven't created p_df yet.
            # Let's create p_df earlier
            p_df = df_data[(df_data['player_id'] == pid)].copy()
            
            p_team = 'UNK'
            if not p_df.empty and 'team_id' in p_df.columns:
                 # Get most common team_id
                 tid_mode = p_df['team_id'].mode()
                 if not tid_mode.empty:
                     tid = int(tid_mode[0])
                     p_team = t_map.get(tid, 'UNK')
            
            if p_team == 'UNK':
                 # Fallback to stats if available (unlikely based on debug)
                 team_counts = {}
                 for s in data['stats']:
                     t = s.get('team', 'UNK')
                     team_counts[t] = team_counts.get(t, 0) + 1
                 if team_counts:
                     p_team = max(team_counts, key=team_counts.get)

            # Output Dir
            out_dir_team = os.path.join(out_dir_base, f'{season}/{p_team}')
            os.makedirs(out_dir_team, exist_ok=True)
            
            # Relative Map & Plot
            team_map = data['grid_team']
            other_map = data['grid_other']
            
            
            # Plot Relative
            rel_path = os.path.join(out_dir_team, f"{pid_int}_relative.png")
            if team_map is not None and league_map is not None:
                try:
                    combined_rel, rel_off_pct, rel_def_pct, rel_off_60, rel_def_60 = compute_relative_map(
                        team_map, league_map, seconds, other_map, seconds, 
                        league_baseline_right=league_map_right
                    )
                    
                    # Plotting code adapted from analyze.players
                    fig, ax = plt.subplots(figsize=(10, 5))
                    draw_rink(ax=ax)
                    from matplotlib.colors import SymLogNorm
                    norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-0.0006, vmax=0.0006, base=10)
                    extent = (-100.5, 100.5, -42.5, 42.5) # Approximate
                    
                    cmap = plt.get_cmap('RdBu_r')
                    try: cmap.set_bad(color=(1,1,1,0)) 
                    except: pass
                    
                    m = np.ma.masked_invalid(combined_rel)
                    im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
                    
                    # Summary Text
                    txt_props = {
                        'home_xg': xg_for,
                        'away_xg': xg_ag,
                        'have_xg': True,
                        'home_goals': total_stats.get('team_goals', 0),
                        'away_goals': total_stats.get('other_goals', 0),
                        'home_attempts': total_stats.get('team_attempts', 0),
                        'away_attempts': total_stats.get('other_attempts', 0),
                        'rel_off_pct': rel_off_pct,
                        'rel_def_pct': rel_def_pct,
                        'home_shot_pct': 0, 'away_shot_pct': 0 # Calc below
                    }
                    
                    tot_att = txt_props['home_attempts'] + txt_props['away_attempts']
                    if tot_att > 0:
                        txt_props['home_shot_pct'] = 100.0 * txt_props['home_attempts'] / tot_att
                        txt_props['away_shot_pct'] = 100.0 * txt_props['away_attempts'] / tot_att

                    add_summary_text(
                        ax=ax, stats=txt_props, main_title=pname, is_season_summary=True,
                        team_name=p_team, full_team_name=pname, filter_str="5v5"
                    )
                    ax.axis('off')
                    fig.savefig(rel_path, dpi=100, bbox_inches='tight') # Reduced DPI for speed/size?
                    plt.close(fig)
                    
                    stats_accumulator.append({
                        'player_id': pid, 'xg_for_60': xg_for_60, 'xg_against_60': xg_ag_60,
                        'games_played': len(data['stats']), 'name': pname, 'team': p_team
                    })
                    
                    # Update Manifest
                    manifest[str(pid)] = {
                        'games_played': int(row['games_played']),
                        'last_game_id': int(row['last_game_id'])
                    }

                except Exception as e:
                    print(f"Error plotting relative for {pname}: {e}")

            # Plot Raw Map with Shots (Legacy Requirement)
            # We filter df_data for this player
            try:
                p_out_path = os.path.join(out_dir_team, f"{pid_int}_map.png")
                # Need just the shots df for this player
                # p_df already created above
                if p_df.empty:
                    print(f"DEBUG: No raw events for {pid_int} in filtered df_data (len={len(df_data)})")
                else: 
                     print(f"Plotting raw map for {pid_int} -> {p_out_path}")
                
                # Custom styles for raw plot consistency
                custom_styles = {
                    'goal': {'marker': 'D', 'size': 80, 'team_color': 'green', 'away_color': 'green', 'zorder': 10},
                    'shot-on-goal': {'marker': 'o', 'size': 30, 'team_color': 'cyan', 'away_color': 'cyan'},
                    'missed-shot': {'marker': 'x', 'size': 30, 'team_color': 'cyan', 'away_color': 'cyan'},
                    'blocked-shot': {'marker': '^', 'size': 30, 'team_color': 'cyan', 'away_color': 'cyan'}
                }

                plot_events(
                    p_df, 
                    events_to_plot=['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'],
                    event_styles=custom_styles,
                    out_path=p_out_path, 
                    title=f"{pname} - {p_team}", 
                    summary_stats=txt_props,
                    heatmap_split_mode='home_away', # Player raw map usually just shows their shots
                    plot_kwargs={'is_season_summary': True, 'filter_str': '5v5', 'team_for_heatmap': p_team},
                    return_heatmaps=False
                )
                plt.close('all')
                plt.close('all')
            except Exception as e:
                pass


        # Memory cleanup after chunk
        gc.collect()

    # Save Manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
        
    print("Done.")

if __name__ == "__main__":
    run_analysis()
