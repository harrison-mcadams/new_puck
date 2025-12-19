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
from puck.plot import add_summary_text, plot_events, plot_relative_map
from puck.analyze import compute_relative_map, league, generate_player_scatter_plots

def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='20252026')
    parser.add_argument('--vmax', type=float, default=None, help='Global colorbar limit')
    parser.add_argument('--scan-limit', action='store_true', help='Scan for global max limit instead of plotting')
    args = parser.parse_args()
    season = args.season
    global_vmax = args.vmax
    scan_limit = args.scan_limit
    
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
    
    # Player -> Team ID Map (Mode)
    # Determine primary team for each player for summary
    print("Building Player-Team Map...")
    try:
        # Optimization: Only needed columns
        pdf = df_data[['player_id', 'team_id']].dropna()
        # Mode is expensive on large group. Just taking last? 
        # Or take max count.
        # ptp = pdf.groupby('player_id')['team_id'].agg(lambda x: x.value_counts().index[0])
        # Faster:
        ptp = pdf.groupby('player_id')['team_id'].apply(lambda x: x.value_counts().index[0])
        pid_tid_map = ptp.to_dict()
    except Exception as e:
        print(f"Warning: Failed to build player-team map: {e}")
        pid_tid_map = {}
    
    # Augment with analysis/teams.json
    teams_json_path = os.path.join(config.ANALYSIS_DIR, 'teams.json')
    if os.path.exists(teams_json_path):
        try:
            with open(teams_json_path, 'r') as f:
                teams_data = json.load(f)
                for t in teams_data:
                    if t.get('id') and t.get('abbr'):
                        tid = int(t['id'])
                        if tid not in t_map:
                            t_map[tid] = t['abbr']
        except Exception: pass

    # Drop heavy columns to save RAM
    # We only need coordinates, IDs, and event metadata for plotting/filtering.
    drop_cols = [
        'event_description', 'game_date', 'venue', 'venue_id', 
        'period_time', 'period_time_remaining', 'original_event_type',
        'strength_state', 'strength_code', 'event_idx'
    ]
    for c in drop_cols:
        if c in df_data.columns:
            df_data.drop(columns=[c], inplace=True)
            
    # Convert string columns to category if helpful (high cardinality strings like 'event_type' are fine, but 'game_state' is low cardinality)
    # This can save significant RAM
    cat_cols = ['game_state', 'event_type', 'home_abb', 'away_abb', 'period_type']
    for c in cat_cols:
        if c in df_data.columns:
            df_data[c] = df_data[c].astype('category')

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

    # ---------------------------------------------------------
    # NEW ARCHITECTURE: Single Pass Loading -> Percentiles -> Plotting
    # ---------------------------------------------------------
    
    # 1. Initialize Aggregation Containers
    # storage for accumulated grids and stats blocks
    # keys: pid (int)
    global_agg = {pid: {'grid_team': None, 'grid_other': None, 'stats': []} for pid in pids_to_process}
    pids_set = set(pids_to_process)

    print(f"Starting Single-Pass Data Loading for {len(pids_to_process)} players...")
    
    # Iterate all available partial files (Games)
    total_files = len(game_path_map)
    for idx, (gid, path) in enumerate(game_path_map.items()):
        if idx % 20 == 0:
            print(f"Loading game {idx}/{total_files}...", end='\r')
            
        try:
            with np.load(path) as data:
                if 'empty' in data: continue
                
                # Check which players from our target list are in this game
                # The NPZ file has keys like "p_{pid}_grid_team"
                # We can iterate keys or inspect manifest inside npz if it existed.
                # Inspecting keys is safer.
                # Optimization: The keys are strings. 
                # Extract PIDs from keys starting with "p_"
                
                # Naive scan of keys is fast enough for 1200 games?
                # Keys: p_8478402_grid_team, ... 
                # Let's extract PIDs present in this file
                # Or iterating our pids_set?
                # Iterating keys is better (usually ~40 players per game)
                
                present_pids = set()
                for k in data.files:
                    if k.startswith('p_') and k.endswith('_stats'):
                        # format: p_{pid}_stats
                        parts = k.split('_')
                        if len(parts) >= 3:
                            try:
                                pid = int(parts[1])
                                present_pids.add(pid)
                            except: pass
                            
                # Intersect with target list
                targets_in_game = present_pids.intersection(pids_set)
                
                for pid in targets_in_game:
                    # Load and Accumulate
                    entry = global_agg[pid]
                    
                    # Grids
                    # handle team
                    k_tm = f"p_{pid}_grid_team"
                    if k_tm in data:
                        g = data[k_tm].astype(np.float32)
                        if entry['grid_team'] is None: entry['grid_team'] = g
                        else: entry['grid_team'] += g
                    
                    # handle other
                    k_ot = f"p_{pid}_grid_other"
                    if k_ot in data:
                        g = data[k_ot].astype(np.float32)
                        if entry['grid_other'] is None: entry['grid_other'] = g
                        else: entry['grid_other'] += g
                        
                    # Stats
                    k_st = f"p_{pid}_stats"
                    if k_st in data:
                        s_str = str(data[k_st])
                        try:
                            s_dict = json.loads(s_str)
                            entry['stats'].append(s_dict)
                        except: pass
                        
        except Exception:
            pass

    print("Data Loading Complete. Computing Aggregates...")

    # 2. Compute Aggregates & Build DataFrame for Percentiles
    summary_data = [] # List of dicts
    
    for pid in pids_to_process:
        data = global_agg[pid]
        if not data['stats']: continue
        
        # Sum Stats
        total_stats = {}
        keys_to_sum = ['team_xgs', 'other_xgs', 'team_goals', 'other_goals', 'team_attempts', 'other_attempts', 'team_seconds']
        for s in data['stats']:
            for k in keys_to_sum:
                total_stats[k] = total_stats.get(k, 0.0) + s.get(k, 0.0)
                
        seconds = total_stats.get('team_seconds', 0.0)
        if seconds <= 60: # Filter low TOI (< 1 min)
            continue
            
        xg_for = total_stats.get('team_xgs', 0.0)
        xg_ag = total_stats.get('other_xgs', 0.0)
        xg_for_60 = (xg_for / seconds) * 3600
        xg_ag_60 = (xg_ag / seconds) * 3600
        
        # Determine Team (Mode of team_id in stats)
        # Use pid_tid_map built earlier
        tid = pid_tid_map.get(pid, -1)
        team_abbr = t_map.get(tid, 'UNK')
        
        # Fallback to stats if needed (unlikely if df_data covered it)
        if team_abbr == 'UNK':
             tids = [s.get('team_id') for s in data['stats'] if 'team_id' in s]
             if tids:
                 tid = max(set(tids), key=tids.count)
                 team_abbr = t_map.get(tid, 'UNK')
             
        # Store for DataFrame
        summary_data.append({
            'player_id': pid,
            'player_name': pid_name_map.get(pid, f"Player {pid}"), # Added early
            'team': team_abbr, # Added
            'games_played': len(data['stats']), # Added for filtering
            'xg_for_60': xg_for_60,
            'xg_ag_60': xg_ag_60,
            'stats': total_stats,
            'seconds': seconds
        })
        
    if not summary_data:
        print("No players found with sufficient data.")
        return

    # Create DataFrame
    df_sum = pd.DataFrame(summary_data)
    
    # Calculate Percentiles (Rank within this set of players)
    # Note: This is rank among "processed players", which might be the whole league if running full update.
    # If running partial update, this rank is local to the batch. 
    # Ideal: Load EXISTING manifest stats to rank against? 
    # Simpler: Just rank within current batch. 
    # For a full run, this is correct.
    from scipy.stats import percentileofscore
    
    print("Calculating Percentiles...")
    df_sum['off_pctile'] = df_sum['xg_for_60'].rank(pct=True) * 100
    # Defense: Lower xG against is better. So invert rank.
    df_sum['def_pctile'] = df_sum['xg_ag_60'].rank(ascending=False, pct=True) * 100
    
    # Map back to dict for generic lookup
    pct_map = df_sum.set_index('player_id')[['off_pctile', 'def_pctile']].to_dict('index')

    # SAVE SUMMARY (New Feature for Verification)
    summary_out_path = os.path.join(out_dir_base, season, 'player_summary_5v5.json')
    try:
        # DF to records
        # df_sum contains 'stats' which is a dict, so plain to_json might be messy if not careful,
        # but to_dict('records') is safe for json.dump
        # df_sum has columns: player_id, xg_for_60, xg_ag_60, stats, seconds, off_pctile, def_pctile
        # We might want to flatten 'stats' or just dump as is.
        # Let's dump the list of dicts that we built + adding pctiles
        
        # Merge pctiles into summary_data list
        for item in summary_data:
            pid = item['player_id']
            if pid in pct_map:
                item['off_pctile'] = pct_map[pid]['off_pctile']
                item['def_pctile'] = pct_map[pid]['def_pctile']
                # Add player name/team for easier debug
                item['player_name'] = pid_name_map.get(pid, f"P{pid}")
                
        with open(summary_out_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved player summary to {summary_out_path}")
        
        # Generate Player Scatter Plots
        try:
            scatter_out_dir = os.path.join(out_dir_base, season)
            os.makedirs(scatter_out_dir, exist_ok=True)
            print("Generating Player Scatter Plots...")
            generate_player_scatter_plots(summary_data, scatter_out_dir)
        except Exception as e:
            print(f"Failed to generate player scatter plots: {e}")
        
    except Exception as e:
        print(f"Warning: Failed to save player summary json: {e}")

    # 3. Plotting Loop
    print(f"Generating Plots for {len(df_sum)} players...")
    
    processed_pids = set()
    manifest_updates = {}
    
    global_scan_max = 0.0
    
    # Config for GC
    gc_counter = 0
    gc_freq = config.GC_FREQUENCY if hasattr(config, 'GC_FREQUENCY') else 50
    
    for idx, (index, row) in enumerate(df_sum.iterrows()):
        if idx % 50 == 0:
             if scan_limit:
                 print(f"Scanning Player {idx}/{len(df_sum)}... (Max: {global_scan_max:.4f})", end='\r')
             else:
                 print(f"Generating Plots: Player {idx}/{len(df_sum)}...", end='\r')
        pid = int(row['player_id'])
        pid_int = pid
        
        # Determine Team
        pname = pid_name_map.get(pid, f"Player {pid}")
        
        # Team Determination logic (Re-use)
        # We need to peek at stats or use df_data
        # df_data is still available
        p_df = df_data[df_data['player_id'] == pid].copy()
        p_team = 'UNK'
        if not p_df.empty and 'team_id' in p_df.columns:
             tid_mode = p_df['team_id'].mode()
             if not tid_mode.empty:
                 tid = int(tid_mode[0])
                 p_team = t_map.get(tid, 'UNK')
                 
        if p_team == 'UNK':
             # stats fallback
             team_counts = {}
             for s in global_agg[pid]['stats']:
                 t = s.get('team', 'UNK')
                 team_counts[t] = team_counts.get(t, 0) + 1
             if team_counts:
                 p_team = max(team_counts, key=team_counts.get)
                 
        # Output Dir
        out_dir_team = os.path.join(out_dir_base, f'{season}/{p_team}')
        os.makedirs(out_dir_team, exist_ok=True)
        
        # Data
        agg_entry = global_agg[pid]
        total_stats = row['stats']
        seconds = row['seconds']
        xg_for_60 = row['xg_for_60']
        xg_ag_60 = row['xg_ag_60']
        
        off_p = pct_map[pid]['off_pctile']
        def_p = pct_map[pid]['def_pctile']

        # Relative Map
        team_map = agg_entry['grid_team']
        other_map = agg_entry['grid_other']
        
        rel_path = os.path.join(out_dir_team, f"{pid}_relative.png")
        
        if team_map is not None and league_map is not None:
            try:
                combined_rel, rel_off_pct, rel_def_pct, rel_off_60, rel_def_60 = compute_relative_map(
                    team_map, league_map, seconds, other_map, seconds, 
                    league_baseline_right=league_map_right
                )
                
                # Check 99.5th Percentile for Limits
                # Use same masking as plotting
                # Neutral zone mask (approx)
                # combined_rel shape is typically (85, 200) or similar
                # Assuming standard grid
                cols = combined_rel.shape[1]
                xs = np.linspace(-100, 100, cols)
                mask_x = np.abs(xs) < 25
                mask = np.tile(mask_x, (combined_rel.shape[0], 1))
                processed_grid_ma = np.ma.masked_where(mask, combined_rel)
                
                # Aggressive saturation
                p80 = np.nanpercentile(np.abs(processed_grid_ma.filled(np.nan)), 80.0)
                if not np.ma.is_masked(p80) and p80 > 0:
                     if p80 > global_scan_max:
                          global_scan_max = p80
                
                if scan_limit:
                     # Skip plotting
                     continue
                
                # Plot using Shared Routine
                # Debug map integrity
                # v_min = np.nanmin(combined_rel)
                # v_max = np.nanmax(combined_rel)
                # print(f"DEBUG {pname}: Range [{v_min:.5f}, {v_max:.5f}]")
                
                minutes = seconds / 60.0
                display_cond = f"5v5 | {minutes:.1f} min"
                
                txt_props = {
                    'home_xg': total_stats.get('team_xgs', 0),
                    'away_xg': total_stats.get('other_xgs', 0),
                    'have_xg': True,
                    'home_goals': int(total_stats.get('team_goals', 0)),
                    'away_goals': int(total_stats.get('other_goals', 0)),
                    'home_attempts': int(total_stats.get('team_attempts', 0)),
                    'away_attempts': int(total_stats.get('other_attempts', 0)),
                    'team_xg_per60': xg_for_60,
                    'other_xg_per60': xg_ag_60,
                    'rel_off_pct': rel_off_pct,
                    'rel_def_pct': rel_def_pct,
                    'off_percentile': off_p,     # Added!
                    'def_percentile': def_p,     # Added!
                    'home_shot_pct': 0, 'away_shot_pct': 0
                }
                
                tot_att = txt_props['home_attempts'] + txt_props['away_attempts']
                if tot_att > 0:
                    txt_props['home_shot_pct'] = 100.0 * txt_props['home_attempts'] / tot_att
                    txt_props['away_shot_pct'] = 100.0 * txt_props['away_attempts'] / tot_att
                
                # PLOT
                fig, ax = plt.subplots(figsize=(10, 6))
                im = plot_relative_map(
                    ax=ax,
                    rel_grid=combined_rel,
                    title=pname,
                    stats=txt_props,
                    team_name=p_team,
                    full_team_name=pname, 
                    cond=display_cond,
                    mask_neutral_zone=True,
                    vmax=global_vmax
                )
                # Background Fix
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # Colorbar Sizing
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                
                import matplotlib.ticker as ticker
                cbar = fig.colorbar(im, cax=cax)
                cbar.locator = ticker.FixedLocator([-0.02, -0.01, 0, 0.01, 0.02])
                cbar.update_ticks()
                cbar.set_label('Excess xG/60 (per 100 sq ft)', rotation=270, labelpad=15)
                
                # Force Axis OFF
                ax.axis('off')
                ax.set_frame_on(False)
                
                fig.savefig(rel_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error plotting relative map for {pname}: {e}")
                
            # Raw Plot
                p_out_path = os.path.join(out_dir_team, f"{pid}_map.png")
                # Need shots DF. 
                # p_df is already filtered and ready.
                
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
                    summary_stats=txt_props, # Use same enriched stats
                    heatmap_split_mode='home_away',
                    plot_kwargs={'is_season_summary': True, 'filter_str': display_cond, 'team_for_heatmap': p_team},
                    return_heatmaps=False
                )
                plt.close('all')
                
                # Update Manifest accumulator
                manifest_updates[str(pid)] = {
                    'games_played': len(agg_entry['stats']),
                    'last_game_id': 0 # Simplification or calc
                }
                
            except Exception as e:
                print(f"Error processing {pname}: {e}")
                
        # --- MEMORY OPTIMIZATION ---
        # Clear specific player data from accumulators instantly
        global_agg[pid] = None 
        del agg_entry
        del p_df
        
        # Periodic GC
        gc_counter += 1
        if gc_counter >= gc_freq:
            gc.collect()
            gc_counter = 0
            # print(f"DEBUG: GC Collected at player {idx}")
        
    # Update Manifest
    manifest.update(manifest_updates)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
        
    print("Done.")
    if scan_limit:
        print(f"SCAN COMPLETE. Max 80th Percentile (Saturated): {global_scan_max}")

if __name__ == "__main__":
    run_analysis()
