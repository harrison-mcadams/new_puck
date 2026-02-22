"""
generate_mixed_heatmaps.py

Generates Heatmaps (Absolute & Relative) for all teams in 2025-2026
using the Mixed Effects xG Model.
Correctly handles Special Teams perspective (5v4 = Team Advantage, 4v5 = Team Disadvantage).
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import percentileofscore
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
import joblib
import pickle

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects, fit_nested_xgs, fit_xgs
from puck.plot import plot_relative_map, plot_events, add_summary_text
from puck.rink import draw_rink
from puck.analyze import generate_scatter_plot
from puck import config

def init_stats_bucket():
    return {
        'grid_for': np.zeros((85, 200)),
        'grid_against': np.zeros((85, 200)),
        'seconds': 0.0, 'xg_for': 0.0, 'xg_against': 0.0,
        'goals_for': 0, 'goals_against': 0,
        'attempts_for': 0, 'attempts_against': 0,
        'games': set()
    }

def main():
    season = "20252026"
    print(f"--- Generating Mixed Effects Heatmaps for {season} ---")

    # 1. Load 2025-2026 Data
    print("Loading Data...")
    data_path = Path("data/20252026.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # Fallback to concatenate
        data_dir = Path("data/20252026")
        files = list(data_dir.glob("*.csv"))
        if not files:
            print("No data found!")
            return
        df = pd.concat([pd.read_csv(f) for f in files])
    
    # Preprocess
    df = fit_nested_xgs.preprocess_features(df)
    

    # Enrich (Handedness etc)
    df = fit_xgs.enrich_data_with_bios(df)
    
    # Enrich Team Name
    def get_team_name(row):
        tid = row['team_id']
        hid = row['home_id']
        aid = row['away_id']
        try:
            if float(tid) == float(hid): return row['home_abb']
            if float(tid) == float(aid): return row['away_abb']
        except: pass
        return "UNKNOWN"

    if 'team_name' not in df.columns:
        if 'home_abb' in df.columns:
            df['team_name'] = df.apply(get_team_name, axis=1)
        else:
            df['team_name'] = df['team_id'].astype(str)
            
    df = df[df['team_name'] != "UNKNOWN"].copy()
    print(f"Data Loaded: {len(df)} rows")
    
    # Pre-compute Global Columns
    if 'period_seconds' not in df.columns:
        def time_to_sec(x):
            try:
                m, s = x.split(':')
                return int(m)*60 + int(s)
            except: return 0
        df['period_seconds'] = df['period_time'].apply(time_to_sec)
        
    if 'total_seconds' not in df.columns:
        df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_seconds']

    # Ensure Context Features exist (if raw data didn't have them)
    # We sort by Game -> Period -> Time to compute deltas
    # Only if essential columns missing
    if 'time_since_last_event' not in df.columns:
        print("  Computing Context Features (Deltas)...")
        # Ensure we have time seconds
        if 'period_seconds' not in df.columns and 'period_time' in df.columns:
            # period_time usually mm:ss
            def time_to_sec(x):
                try:
                    m, s = x.split(':')
                    return int(m)*60 + int(s)
                except: return 0
            df['period_seconds'] = df['period_time'].apply(time_to_sec)
            
        # Sort
        df.sort_values(['game_id', 'period', 'period_seconds'], inplace=True)
        
        # Shift
        df['last_event_time'] = df.groupby('game_id')['period_seconds'].shift(1)
        df['last_event_period'] = df.groupby('game_id')['period'].shift(1)
        
        # Time Delta (handle period changes? assume 0 if diff period for simplicity, or just treat within period)
        # Actually usually time_since_last_event resets on period start.
        df['time_since_last_event'] = df['period_seconds'] - df['last_event_time']
        df.loc[df['period'] != df['last_event_period'], 'time_since_last_event'] = 0
        df['time_since_last_event'] = df['time_since_last_event'].fillna(0).clip(lower=0)
        
        # Last Event Type/Team
        df['last_event_type'] = df.groupby('game_id')['event'].shift(1).fillna('None')
        df['last_event_team'] = df.groupby('game_id')['team_name'].shift(1).fillna('None')
        
        # Spatial Deltas (if x_adj/y_adj exist)
        if 'x_adj' in df.columns:
            df['last_x'] = df.groupby('game_id')['x_adj'].shift(1).fillna(0)
            df['last_y'] = df.groupby('game_id')['y_adj'].shift(1).fillna(0)
            df['dist_from_last_event'] = np.sqrt((df['x_adj'] - df['last_x'])**2 + (df['y_adj'] - df['last_y'])**2)
            df.loc[df['period'] != df['last_event_period'], 'dist_from_last_event'] = 0
            
            # Speed
            df['speed_from_last_event'] = df['dist_from_last_event'] / df['time_since_last_event']
            df['speed_from_last_event'] = df['speed_from_last_event'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Angle Change? Complex, requires velocity vector. Placeholder 0.
            df['angle_change_last_event'] = 0.0
            
        # Is Rebound? (Shot within 3s of another shot)
        # df['last_event_type'] might include 'shot-on-goal', 'blocked-shot', 'missed-shot'
        shot_events = ['shot-on-goal', 'blocked-shot', 'missed-shot', 'goal']
        df['is_rebound'] = (df['time_since_last_event'] <= 3.0) & (df['last_event_type'].isin(shot_events))
        
        # Is Rush? (Speed > threshold? or Time < threshold & Dist > threshold?)
        # Simple proxy:
        df['is_rush'] = (df['speed_from_last_event'] > 20.0).astype(int)

    # Grid Specs
    BIN_X = np.linspace(-100, 100, 201) 
    BIN_Y = np.linspace(-42.5, 42.5, 86)
    
    out_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # GRAND STATS BUCKET
    # Structure: grand_stats[team][condition]
    # conditions: '5v5', '5v4' (Advantage), '4v5' (Disadvantage)
    grand_stats = {} 
    
    processed_states = ['5v5', '5v4', '4v5'] 
    
    # Define features to use for Mixed Effects (consistent with events bank)
    # These must match what we export to events_bank for simulation!
    me_features = [
        'x_adj', 'y_adj', 'distance', 'angle_deg',
        'time_since_last_event', 'angle_change_last_event', 
        'speed_from_last_event', 'dist_from_last_event',
        'rebound_angle_change', 'rebound_time_diff', 'rebound_speed', 'rebound_dist_change'
    ]
    # Filter to what exists in df
    me_features = [f for f in me_features if f in df.columns]
    print(f"  Mixed Effects Feature Set ({len(me_features)}): {me_features}")

    for state in ['5v5', '5v4', '4v5']:
        print(f"\n=== Processing Mixed Effects Model and Events for Global State: {state} ===")
        
        # Filter for current state
    # --- PRE-COMPUTE CLEAN INTERVALS ---
    print("Pre-computing Clean Intervals (Consensus Filtering)...")
    
    # Map: gid -> state -> list of (s,e, duration)
    clean_intervals_map = {} 
    
    # We need to compute this for ALL games
    all_game_ids = df['game_id'].unique()
    
    # Prepare global lookup for events (time sorted)
    # We already sorted df by game_id, period, seconds
    
    def get_clean_intervals_for_game(gid):
        g_df = df[df['game_id'] == gid]
        if g_df.empty: return None
        
        home_abb = g_df['home_abb'].iloc[0]
        away_abb = g_df['away_abb'].iloc[0]
        
        game_res = {}
        
        # Check standard states
        for state in ['5v5', '5v4', '4v5']:
            try:
                from puck import timing
                cond = {'game_state': [state], 'is_net_empty': [0]}
                raw_intervals = timing.get_game_intervals_cached(gid, season, cond)
                
                valid_ints = []
                total_dur = 0.0
                
                for s, e in raw_intervals:
                    dur = e - s
                    if dur <= 0: continue
                    
                    # Consensus Check
                    # 5v4/4v5: If > 50% events are 5v5, discard.
                    # 5v5: If > 50% events are 5v4/4v5, discard? Usually we trust 5v5 shifts more.
                    # For now, apply STRICT Consensus to Special Teams.
                    
                    
                    # REMOVED: Consensus Check for Phantom Intervals
                    # Empirical analysis showed this discards valid EDM PP data
                    # because events are systematically mislabeled as 5v5.
                    is_phantom = False
                    # if state in ['5v4', '4v5']: ...

                    
                    if not is_phantom:
                        valid_ints.append((s, e))
                        total_dur += dur
                        
                game_res[state] = {
                    'intervals': valid_ints,
                    'seconds': total_dur
                }
            except:
                game_res[state] = {'intervals': [], 'seconds': 0.0}
                
        return (gid, game_res)

    # Parallelize Interval Computation
    interval_results = Parallel(n_jobs=-1, verbose=1)(delayed(get_clean_intervals_for_game)(gid) for gid in all_game_ids)
    
    for res in interval_results:
        if res:
            clean_intervals_map[res[0]] = res[1]

    # --- MAIN STATE LOOP ---
    
    output_states = ['5v5', '5v4', '4v5']
    grand_stats = {} # team -> state -> bucket
    
    # Load Unified Mixed Effects Model Once
    model = None
    model_path = Path("analysis/xgs/joint_mixed_effects.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Loaded unified mixed effects model from {model_path}")
    else:
        print(f"Warning: Unified model not found at {model_path}. Predict xG will use 0.0.")

    for state in output_states:
        print(f"\nProcessing State: {state} ...")
        
        # 2. Select Events by CLEAN TIME (Strictly Trust Shift Chart)
        print(f"  Selecting Valid Events for {state}...")
        
        # Optimization: Just return indices
        def get_valid_indices(gid):
            if gid not in clean_intervals_map: return []
            info = clean_intervals_map[gid].get(state)
            if not info or not info['intervals']: return []
            
            g_df = df[df['game_id'] == gid]
            intervals = info['intervals']
            
            t_vals = g_df['total_seconds'].values
            
            # Check against intervals
            mask = np.zeros(len(g_df), dtype=bool)
            for s, e in intervals:
                 mask |= ((t_vals >= s) & (t_vals < e))
                 
            mask &= (g_df['is_net_empty'] == 0).values
            
            # REMOVED: Consensus Filtering (is_valid ratio check)
            # We assume Shift Chart is Ground Truth.
            # Systematic mislabeling (5v5 events in 5v4 time) requires this.
            
            return g_df.index[mask].tolist()

        indices_chunks = Parallel(n_jobs=-1, verbose=1)(delayed(get_valid_indices)(gid) for gid in all_game_ids)
        flat_indices = [i for chunk in indices_chunks for i in chunk]
        
        df_state = df.loc[flat_indices].copy()
        print(f"  Selected {len(df_state)} events for {state}")
        
        # 3. Predict xG (using Unified Model)
        if model is not None:
             print(f"  Predicting xG for {state}...")
             # Need opp_team_name for Dual Model
             def get_opp_name(row):
                 if row['team_name'] == row['home_abb']: return row['away_abb']
                 return row['home_abb']
             df_state['opp_team_name'] = df_state.apply(get_opp_name, axis=1)
             
             # Map off/def for StateMixedEffectsModel
             df_pred = df_state.copy()
             df_pred['off_team_name'] = df_pred['team_name']
             df_pred['def_team_name'] = df_pred['opp_team_name']
             
             probs = model.predict_proba(df_pred)[:, 1]
             df_state['xgs'] = probs
        else:
             # For 5v5, we don't have a specific model trained in this loop.
             # If we wanted 5v5 xG, we'd need to train/load a 5v5 model here.
             # For now, setting to 0.0 for states without a model.
             df_state['xgs'] = 0.0 

        # 5. Process Games (Aggregation)
        game_ids_with_events = df_state['game_id'].unique()
        
        # We also need to process games that have NO events but HAVE time (valid empty intervals).
        # So we should iterate ALL clean_intervals_map keys that have duration > 0 for this state.
        
        relevant_gids = set(game_ids_with_events)
        for gid, info in clean_intervals_map.items():
            if info.get(state, {}).get('seconds', 0) > 0:
                relevant_gids.add(gid)
        
        print(f"  Aggregating stats for {len(relevant_gids)} games...")

        def process_game(gid):
             try:
                # Events for this game in this state
                g_ev = df_state[df_state['game_id'] == gid]
                
                # Metadata
                # We need home/away teams.
                if g_ev.empty:
                     # Access main df global?
                     # Safer: extract metadata from map if possible, or fetch from df.
                     # We can fetch 1 row from df (which is indexed/fast)
                     row = df[df['game_id'] == gid].iloc[0]
                     home_team = row['home_abb']
                     away_team = row['away_abb']
                else:
                     home_team = g_ev['home_abb'].iloc[0]
                     away_team = g_ev['away_abb'].iloc[0]
                     
                seconds = clean_intervals_map[gid][state]['seconds']
                
                res = {
                    'game_id': gid,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_stats': None,
                    'away_stats': None
                }

                for role, team, opp in [('home', home_team, away_team), ('away', away_team, home_team)]:
                    if g_ev.empty:
                        events = pd.DataFrame()
                        opp_events = pd.DataFrame()
                    else:
                        events = g_ev[g_ev['team_name'] == team]
                        opp_events = g_ev[g_ev['team_name'] == opp]
                    
                    stats = {
                        'seconds': seconds,
                        'xg_for': events['xgs'].sum() if not events.empty else 0.0,
                        'xg_against': opp_events['xgs'].sum() if not opp_events.empty else 0.0,
                        # FIX: Count goals using RAW event labels, not interval-filtered df
                        # This ensures goals labeled 5v4/5v3/5v6 are counted even if they
                        # fall slightly outside the Shift Chart interval boundaries.
                        'goals_for': 0,  # Will be computed below
                        'goals_against': 0,  # Will be computed below
                        'attempts_for': len(events),
                        'attempts_against': len(opp_events),
                        'grid_for': np.zeros((85, 200)),
                        'grid_against': np.zeros((85, 200))
                    }
                    
                    # Count goals from RAW df (not interval-filtered df_state)
                    # For 5v4 state: count goals labeled 5v4, 5v3, 5v6 (all PP variants)
                    # For 4v5 state: count goals labeled 4v5, 3v5, 4v6 (all PK variants)
                    g_raw = df[(df['game_id'] == gid) & (df['event'] == 'goal')]
                    
                    # Get team_id for this team
                    if not g_raw.empty:
                        team_id_lookup = df[(df['game_id'] == gid)].iloc[0]
                        if team == team_id_lookup['home_abb']:
                            team_id = team_id_lookup['home_id']
                        else:
                            team_id = team_id_lookup['away_id']
                        
                        if state == '5v4':
                            pp_states = ['5v4', '5v3', '5v6', '6v4', '6v5']  # Home PP variants
                            team_goals = g_raw[(g_raw['team_id'] == team_id) & (g_raw['game_state'].isin(pp_states))]
                            opp_goals = g_raw[(g_raw['team_id'] != team_id) & (g_raw['team_id'].isin([team_id_lookup['home_id'], team_id_lookup['away_id']])) & (g_raw['game_state'].isin(pp_states))]
                        elif state == '4v5':
                            pk_states = ['4v5', '3v5', '4v6', '4v3', '3v4']  # Home PK variants  
                            team_goals = g_raw[(g_raw['team_id'] == team_id) & (g_raw['game_state'].isin(pk_states))]
                            opp_goals = g_raw[(g_raw['team_id'] != team_id) & (g_raw['team_id'].isin([team_id_lookup['home_id'], team_id_lookup['away_id']])) & (g_raw['game_state'].isin(pk_states))]
                        else:  # 5v5
                            team_goals = g_raw[(g_raw['team_id'] == team_id) & (g_raw['game_state'] == '5v5')]
                            opp_goals = g_raw[(g_raw['team_id'] != team_id) & (g_raw['team_id'].isin([team_id_lookup['home_id'], team_id_lookup['away_id']])) & (g_raw['game_state'] == '5v5')]
                        
                        stats['goals_for'] = len(team_goals)
                        stats['goals_against'] = len(opp_goals)
                    
                    if not events.empty:
                        # x starts on right (0..100) -> map to left (-100..0)
                        x_for = -events['x_adj'].abs()
                        y_for = -events['y_adj']
                        H, _, _ = np.histogram2d(y_for, x_for, bins=[BIN_Y, BIN_X], weights=events['xgs'])
                        stats['grid_for'] += H
                        
                    if not opp_events.empty:
                        # x starts on right (0..100) -> stays right (0..100)
                        x_ag = opp_events['x_adj'].abs()
                        y_ag = opp_events['y_adj']
                        H, _, _ = np.histogram2d(y_ag, x_ag, bins=[BIN_Y, BIN_X], weights=opp_events['xgs'])
                        stats['grid_against'] += H
                    
                    if role == 'home': res['home_stats'] = stats
                    else: res['away_stats'] = stats
                
                return res
             except Exception as e:
                return None

        results = Parallel(n_jobs=-1, verbose=1)(delayed(process_game)(gid) for gid in relevant_gids)
                
    
        # Merge Results and Route to Buckets
        print(f"  Merging results for {state}...")
        for res in results:
            if not res: continue
            
            home_team = res['home_team']
            away_team = res['away_team']
            gid = res['game_id']
            h_stats = res['home_stats']
            a_stats = res['away_stats']
            
            # Determine Target Condition for Home/Away based on Global State
            home_cond = None
            away_cond = None
            
            if state == '5v5':
                home_cond = '5v5'
                away_cond = '5v5'
            elif state == '5v4':
                home_cond = '5v4' # Home has 5 = Advantage
                away_cond = '4v5' # Away has 4 = Disadvantage
            elif state == '4v5':
                home_cond = '4v5' # Home has 4 = Disadvantage
                away_cond = '5v4' # Away has 5 = Advantage
                
            # Init Team Buckets if needed
            for t in [home_team, away_team]:
                if t not in grand_stats:
                    grand_stats[t] = {
                        '5v5': init_stats_bucket(),
                        '5v4': init_stats_bucket(),
                        '4v5': init_stats_bucket()
                    }
            
            # Accumulate Home
            if home_cond:
                ts = grand_stats[home_team][home_cond]
                s = h_stats
                ts['seconds'] += s['seconds']
                ts['xg_for'] += s['xg_for']
                ts['xg_against'] += s['xg_against']
                ts['goals_for'] += s['goals_for']
                ts['goals_against'] += s['goals_against']
                ts['attempts_for'] += s['attempts_for']
                ts['attempts_against'] += s['attempts_against']
                ts['grid_for'] += s['grid_for']
                ts['grid_against'] += s['grid_against']
                ts['games'].add(gid)
                
            # Accumulate Away
            if away_cond:
                ts = grand_stats[away_team][away_cond]
                s = a_stats
                ts['seconds'] += s['seconds']
                ts['xg_for'] += s['xg_for']
                ts['xg_against'] += s['xg_against']
                ts['goals_for'] += s['goals_for']
                ts['goals_against'] += s['goals_against']
                ts['attempts_for'] += s['attempts_for']
                ts['attempts_against'] += s['attempts_against']
                ts['grid_for'] += s['grid_for']
                ts['grid_against'] += s['grid_against']
                ts['games'].add(gid)

    # --- PLOTTING PHASE ---
    
    # We now have robust stats for each team in each bucket.
    # We iterate the buckets ['5v5', '5v4', '4v5'] to generate plots.
    
    output_states = ['5v5', '5v4', '4v5']
    
    for plot_state in output_states:
        print(f"\n=== Generating Plots for Team Condition: {plot_state} ===")
        
        # 1. Compute League Totals for this Condition
        league_grid_sum = np.zeros((85, 200))
        league_seconds = 0.0
        
        # Collect stats for normalization/percentiles
        all_xgf60 = []
        all_xga60 = []
        
        valid_teams = []
        
        for team, buckets in grand_stats.items():
            s = buckets[plot_state]
            if s['seconds'] < 60: continue # Min 1 min TOI
            
            valid_teams.append(team)
            league_seconds += s['seconds']
            league_grid_sum += (s['grid_for'] + s['grid_against']) # Total xG density
            
            s['xgf60'] = (s['xg_for'] / s['seconds']) * 3600
            s['xga60'] = (s['xg_against'] / s['seconds']) * 3600
            
            all_xgf60.append(s['xgf60'])
            all_xga60.append(s['xga60'])
            
        if league_seconds == 0:
            print(f"  No valid data for {plot_state}")
            continue
            
        league_norm_grid = league_grid_sum / league_seconds
        avg_xgf60 = np.mean(all_xgf60) if all_xgf60 else 0
        avg_xga60 = np.mean(all_xga60) if all_xga60 else 0
        
        # 2. Plot Per Team
        out_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        summary_list = []
        
        for team in valid_teams:
            s = grand_stats[team][plot_state]
            
            # 6a. Compute Relative Grid
            grid_total = s['grid_for'] + s['grid_against']
            team_norm = grid_total / s['seconds']
            
            # Increase smoothing to match daily.py (sigma=6.0)
            team_norm_smooth = gaussian_filter(team_norm, sigma=6.0)
            league_norm_smooth = gaussian_filter(league_norm_grid, sigma=6.0)
            rel_grid = (team_norm_smooth - league_norm_smooth) * 3600 * 100
            
            # 6b. Stats Dictionary
            stats_dict = s.copy()
            if len(all_xgf60) > 0:
                stats_dict['off_percentile'] = percentileofscore(all_xgf60, s['xgf60'])
                stats_dict['def_percentile'] = 100 - percentileofscore(all_xga60, s['xga60'])
            else:
                stats_dict['off_percentile'] = 50; stats_dict['def_percentile'] = 50
                
            stats_dict['rel_off_pct'] = 100 * (s['xgf60'] - avg_xgf60) / avg_xgf60 if avg_xgf60 else 0
            stats_dict['rel_def_pct'] = 100 * (s['xga60'] - avg_xga60) / avg_xga60 if avg_xga60 else 0
            
            tot_att = s['attempts_for'] + s['attempts_against']
            stats_dict['home_shot_pct'] = 100 * s['attempts_for'] / tot_att if tot_att else 0
            stats_dict['away_shot_pct'] = 100 * s['attempts_against'] / tot_att if tot_att else 0
            
            stats_dict['team_xg_per60'] = s['xgf60']
            stats_dict['other_xg_per60'] = s['xga60']
            
            stats_dict['home_goals'] = s['goals_for']
            stats_dict['away_goals'] = s['goals_against']
            stats_dict['home_xg'] = s['xg_for']
            stats_dict['away_xg'] = s['xg_against']
            stats_dict['have_xg'] = True
            stats_dict['team'] = team
            
            summary_list.append(stats_dict)
            
            # PLOT RELATIVE
            fig, ax = plt.subplots(figsize=(10, 6))
            im = plot_relative_map(
                ax=ax, rel_grid=rel_grid,
                title=f"{team} Mixed Relative xG ({plot_state})",
                stats=stats_dict, team_name=team, full_team_name=team,
                cond=plot_state, mask_neutral_zone=True
            )
            fig.patch.set_facecolor('white')
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            out_rel = out_dir / f"{team}_mixed_relative_{plot_state}.png"
            fig.savefig(out_rel, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            # PLOT ABSOLUTE
            fig_abs, ax_abs = plt.subplots(figsize=(10,6))
            draw_rink(ax_abs)
            mask_x = np.abs(np.linspace(-100, 100, 200)) < 25
            mask = np.tile(mask_x, (85, 1))
            abs_grid_view = team_norm_smooth * 3600 * 100 
            abs_grid_ma = np.ma.masked_where(mask, abs_grid_view)
            
            ax_abs.imshow(abs_grid_ma, extent=[-100, 100, -42.5, 42.5], origin='lower', cmap='inferno')
            ax_abs.set_title(f"{team} Mixed xG Density ({plot_state})")
            add_summary_text(ax_abs, stats_dict, f"{team} Mixed xG Density", is_season_summary=True, team_name=team)
            
            out_abs = out_dir / f"{team}_mixed_absolute_{plot_state}.png"
            fig_abs.savefig(out_abs, dpi=120, bbox_inches='tight')
            plt.close(fig_abs)
            
            # SCATTER
            mask_events_state = (df['game_state'] == plot_state) # APPROXIMATE for scatter, ideally split but this is hard for scatter events.
            # Using specific team events for plot_state is tricky if we filtered earlier.
            # We want events where THIS team had condition 'plot_state'.
            # Global '5v5' -> Both 5v5.
            # Global '5v4' -> Home is 5v4. Away is 4v5.
            # Global '4v5' -> Home is 4v5. Away is 5v4.
            
            # Filter main df for scatter events
            # Logic: 
            # If plot_state == '5v4':
            #   (game_state == '5v4' & team == home) OR (game_state == '4v5' & team == away)
            # If plot_state == '4v5':
            #   (game_state == '4v5' & team == home) OR (game_state == '5v4' & team == away)
            # If plot_state == '5v5':
            #   game_state == '5v5' & team == team
            
            cond1 = False
            if plot_state == '5v5':
                cond1 = (df['game_state'] == '5v5') & (df['team_name'] == team)
            elif plot_state == '5v4':
                c1 = (df['game_state'] == '5v4') & (df['team_name'] == team) & (df['home_abb'] == team) # Team is Home & 5v4
                # Or Team is Away and game is 4v5 (Home 4 Away 5) -> Away has 5.
                c2 = (df['game_state'] == '4v5') & (df['team_name'] == team) & (df['away_abb'] == team)
                cond1 = c1 | c2
            elif plot_state == '4v5':
                c1 = (df['game_state'] == '4v5') & (df['team_name'] == team) & (df['home_abb'] == team) # Team is Home & 4v5
                # Or Team is Away and game is 5v4 (Home 5 Away 4) -> Away has 4
                c2 = (df['game_state'] == '5v4') & (df['team_name'] == team) & (df['away_abb'] == team)
                cond1 = c1 | c2
                
            team_events = df[cond1].copy()
            
            if not team_events.empty:
                out_scatter = out_dir / f"{team}_mixed_scatter_{plot_state}.png"
                events_for_plot = team_events.copy()
                events_for_plot['x'] = -events_for_plot['x_adj'].abs()
                events_for_plot['y'] = events_for_plot['y_adj']
                
                plot_events(
                    events=events_for_plot,
                    events_to_plot=['goal', 'shot-on-goal'],
                    title=f"{team} Season Scatter ({plot_state})",
                    out_path=str(out_scatter),
                    summary_stats=stats_dict,
                    rink=True
                )
                plt.close('all')

        # 3. League Scatter
        print(f"  Generating League-Wide Scatter Plot for {plot_state}...")
        try:
            generate_scatter_plot(summary_list, str(out_dir), condition_name=f"Mixed Effects {plot_state}")
            default_scatter = out_dir / "scatter.png"
            if default_scatter.exists():
                new_scatter = out_dir / f"scatter_{plot_state}.png"
                if new_scatter.exists():
                    new_scatter.unlink()
                default_scatter.rename(new_scatter)
                print(f"  Saved {new_scatter}")
        except Exception as e:
            print(f"Failed to generate score plot for {plot_state}: {e}")

    # 4. Export Team Stats Summary
    print("  Exporting Team Stats Summary JSON...")
    serializable_stats = {}
    for team, buckets in grand_stats.items():
        serializable_stats[team] = {}
        for cond, s in buckets.items():
            serializable_stats[team][cond] = {
                'seconds': float(s['seconds']),
                'xg_for': float(s['xg_for']),
                'xg_against': float(s['xg_against']),
                'goals_for': int(s['goals_for']),
                'goals_against': int(s['goals_against']),
                'attempts_for': int(s['attempts_for']),
                'attempts_against': int(s['attempts_against']),
                'games_played': int(len(s['games']))
            }
    
    with open(out_dir / "team_stats_summary.json", 'w') as f:
        json.dump(serializable_stats, f, indent=2)

    # 5. Export Grids (Pickle)
    print("  Exporting Team Grids (Pickle)...")
    # We only need grids, but saving full grand_stats is massive.
    # Let's clean it up or save a specific dict.
    grid_export = {}
    for team, buckets in grand_stats.items():
        grid_export[team] = {}
        for cond, s in buckets.items():
            grid_export[team][cond] = {
                'grid_for': s['grid_for'],       # numpy array 85x200
                'grid_against': s['grid_against'] # numpy array 85x200
            }
            
    with open(out_dir / "team_grids.pkl", 'wb') as f:
        pickle.dump(grid_export, f)

    with open(out_dir / "team_grids.pkl", 'wb') as f:
        pickle.dump(grid_export, f)

    # 6. Export Events Bank (for Matchup Simulation)
    print("  Exporting Events Bank (Pickle)...")
    
    # Ensure opp_team_name exists in global df
    if 'opp_team_name' not in df.columns:
        df['opp_team_name'] = np.where(df['team_name'] == df['home_abb'], df['away_abb'], df['home_abb'])
        
    # Columns for context sampling & Simulation
    # START with Basic Metadata
    bank_cols = [
        'game_id', 'team_name', 'opp_team_name', 'game_state',
        'event', 'period', 'period_seconds', 
        'x_adj', 'y_adj' # spatial
    ]
    
    # ADD All Features required by the Model
    # 1. Mixed Effects Features (Random Slopes)
    print(f"  Adding {len(me_features)} Mixed Effects Features to Events Bank export...")
    bank_cols.extend(me_features)
    
    # 2. Base GLM Features (Fixed Effects)
    # Hardcoded list from inspection to avoid scope/NameError issues
    base_glm_features = [
        'distance', 'angle_deg', 'game_state', 'score_diff', 
        'period_number', 'time_elapsed_in_period_s', 'total_time_elapsed_s',
        'shot_type', 'shoots_catches', 'is_rebound', 'is_rush',
        'rebound_angle_change', 'rebound_time_diff', 
        'last_event_type', 'last_event_time_diff', 
        'dist_from_last_event', 'speed_from_last_event', 'angle_change_last_event',
        'shooter_role'
    ]
    bank_cols.extend(base_glm_features)
    
    # Deduplicate
    bank_cols = list(set(bank_cols))
    
    # Filter only what exists in df
    bank_cols = [c for c in bank_cols if c in df.columns]
    
    events_bank = df[bank_cols].copy()



    # Compress types to save space?
    for c in events_bank.select_dtypes(include=['float64']).columns:
        events_bank[c] = events_bank[c].astype('float32')
        
    events_bank.to_pickle(out_dir / "events_bank.pkl")

    print(f"Done. Maps saved to {out_dir}")

if __name__ == "__main__":
    main()
