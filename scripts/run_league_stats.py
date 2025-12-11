#!/usr/bin/env python3
"""
run_league_stats.py

Aggregates daily partials into league-wide stats and maps.
Generates:
1. League Baseline Grid (.npy)
2. Team Summaries (.json, .csv)
3. Team Plots:
   - Season Summary (Raw Attempts)
   - Relative xG Map (Team vs League)
4. League Scatter Plot

Refactored to support:
- 3 Phases: Aggregate -> Compute Metrics -> Plot
- Full Rink Relative Maps (Offense Left, Defense Right)
- Percentiles and Distribution Stats in text
- Correct Time Display
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import json
from scipy.stats import percentileofscore

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config
from puck import analyze
from puck.plot import plot_events, add_summary_text
from puck.rink import draw_rink
from puck.parse import build_mask # For filtering raw events
from puck.analyze import generate_scatter_plot

def run_league_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='20252026')
    args = parser.parse_args()
    
    season = args.season
    print(f"--- Cached League Analysis for {season} ---")
    
    # Paths
    cache_dir = os.path.join(config.get_cache_dir(season), 'partials')
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return

    print(f"Checking cache dir: {cache_dir}")
    
    
    # Load Season Data (Hybrid approach: needed for raw event plotting)
    # rank_df_path = os.path.join(config.get_data_dir(season), f"{season}_df.csv")
    rank_df_path = os.path.join(config.DATA_DIR, season, f"{season}_df.csv")
    if os.path.exists(rank_df_path):
        df_season = pd.read_csv(rank_df_path)
        # Ensure integer IDs
        for c in ['team_id', 'home_id', 'away_id']:
            if c in df_season.columns:
                df_season[c] = df_season[c].fillna(-1).astype(int)
    else:
        df_season = pd.DataFrame()
        print("Warning: Season CSV not found. Raw plots will be skipped.")

    # Process Conditions
    conditions = ['5v5', '5v4', '4v5']
    
    # Team Mapping
    # Ideally load from API or existing map. Construct from season DF if possible
    t_map = {}
    if not df_season.empty:
        # Construct t_map from home/away columns
        h_map = df_season[['home_id', 'home_abb']].rename(columns={'home_id': 'tid', 'home_abb': 'abb'}).dropna().drop_duplicates()
        a_map = df_season[['away_id', 'away_abb']].rename(columns={'away_id': 'tid', 'away_abb': 'abb'}).dropna().drop_duplicates()
        combined = pd.concat([h_map, a_map]).drop_duplicates(subset='tid')
        t_map = combined.set_index('tid')['abb'].to_dict()
    
    for cond in conditions:
        print(f"Processing {cond}...")
        
        # Files for this condition
        files = [f for f in os.listdir(cache_dir) if f.endswith(f"_{cond}.npz")]
        if not files:
            continue
            
        print(f"  Found {len(files)} game files.")
        
        # Output Dir
        # out_root = os.path.join(config.get_analysis_dir(season), 'league', cond)
        out_root = os.path.join(config.ANALYSIS_DIR, 'league', season, cond)
        os.makedirs(out_root, exist_ok=True)
        
        # Prepare filtered DF for (Raw Plotting)
        df_cond = pd.DataFrame()
        cond_dict = {'5v5': {'game_state': ['5v5'], 'is_net_empty': 0}, 
                     '5v4': {'game_state': ['5v4'], 'is_net_empty': 0}, 
                     '4v5': {'game_state': ['4v5'], 'is_net_empty': 0}}.get(cond)
        if cond_dict and not df_season.empty:
            mask = build_mask(df_season, cond_dict)
            df_cond = df_season[mask].copy()
            for col in ['team_id', 'home_id', 'away_id']:
                if col in df_cond.columns:
                    df_cond[col] = df_cond[col].fillna(-1).astype(int)

        # --- PHASE 1: AGGREGATE ---
        # We need to build Full Rink Grids for each team.
        # Left Side = FOR (Offense)
        # Right Side = AGAINST (Defense) => From Opponent's FOR (Rotated 180)
        
        team_grids = {} # tid -> np.array (Accumulator)
        team_stats = {} # tid -> dict (Accumulator)
        
        # League Accumulator
        league_grid_sum = None # Sum of all team full grids
        league_seconds_sum = 0.0 # Sum of all team seconds
        
        # Standard Grid Shape check
        # We assume all cached grids are same shape
        
        for fname in files:
            try:
                path = os.path.join(cache_dir, fname)
                with np.load(path, allow_pickle=True) as data:
                    if 'empty' in data: continue
                    
                    keys = list(data.keys())
                    
                    # Identify Teams in this file
                    # Keys: team_{tid}_grid_team
                    tids_in_game = []
                    for k in keys:
                        if k.startswith('team_') and k.endswith('_grid_team'):
                            tids_in_game.append(int(k.split('_')[1]))
                    
                    # We expect exactly 2 teams per game usually
                    # But handle whatever is there
                    
                    # First pass: Load 'For' grids and Stats
                    game_grids = {} # tid -> grid
                    
                    for tid in tids_in_game:
                        # Load Stats
                        k_stat = f"team_{tid}_stats"
                        if k_stat in data:
                            if data[k_stat].dtype.kind in {'U', 'S'}:
                                s = json.loads(str(data[k_stat].item()))
                            else:
                                s = json.loads(str(data[k_stat]))
                                
                            if tid not in team_stats:
                                team_stats[tid] = {'team_xgs': 0.0, 'other_xgs': 0.0, 
                                                   'team_seconds': 0.0, 'team_goals': 0, 
                                                   'other_goals': 0, 'team_attempts': 0, 
                                                   'other_attempts': 0, 'n_games': 0}
                            ts = team_stats[tid]
                            ts['team_xgs'] += s.get('team_xgs', 0.0)
                            ts['other_xgs'] += s.get('other_xgs', 0.0)
                            ts['team_seconds'] += s.get('team_seconds', 0.0)
                            ts['team_goals'] += s.get('team_goals', 0)
                            ts['other_goals'] += s.get('other_goals', 0)
                            ts['team_attempts'] += s.get('team_attempts', 0)
                            ts['other_attempts'] += s.get('other_attempts', 0)
                            ts['n_games'] += 1
                        
                        # Load 'For' Grid
                        k_grid = f"team_{tid}_grid_team"
                        if k_grid in data:
                            g = np.nan_to_num(data[k_grid])
                            game_grids[tid] = g
                            
                    # Second pass: Accumulate Full Grids (For + Against)
                    # Against comes from OPPONENT'S 'For' grid, ROTATED.
                    
                    # Identify opponents. If 2 teams, they are opponents.
                    if len(tids_in_game) == 2:
                        t1, t2 = tids_in_game
                        opp_map = {t1: t2, t2: t1}
                    else:
                        # Fallback/Edge case? Ignore cross-fill if not strictly 1v1
                        opp_map = {}
                        
                    for tid in tids_in_game:
                        if tid not in game_grids: continue
                        
                        grid_for = game_grids[tid]
                        
                        # Find Opponent Grid (Against)
                        grid_against = np.zeros_like(grid_for)
                        if tid in opp_map:
                            opp_id = opp_map[tid]
                            if opp_id in game_grids:
                                # Rotate Opponent's For Grid to become This Team's Against Grid
                                # Opponent Shoots Left -> Rotate 180 -> Opponent Shoots Right (Defended by This Team)
                                grid_against = np.rot90(game_grids[opp_id], 2)
                                
                        full_game_grid = grid_for + grid_against
                        
                        # Add to Team Accumulator
                        if tid not in team_grids:
                            team_grids[tid] = full_game_grid.astype(np.float64)
                        else:
                            team_grids[tid] += full_game_grid
                            
                        # Add to League Accumulator
                        if league_grid_sum is None:
                            league_grid_sum = full_game_grid.astype(np.float64)
                        else:
                            league_grid_sum += full_game_grid
                            
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                
        # --- PHASE 2: COMPUTE METRICS ---
        if not team_stats:
            print(f"  No stats found for {cond}.")
            continue
            
        # 1. Total League Seconds (Normalize League)
        # Sum of all team seconds = 2 * Real Time (since we sum both sides)
        total_team_seconds = sum(t['team_seconds'] for t in team_stats.values())
        if total_team_seconds <= 0: continue
        
        # League Average Grid = (Sum of All Team Grids) / Total Team Seconds
        # This gives Rates per second.
        # Note: Sum of Team Grids includes (For + Against) for every team.
        # Basically 2 * (All Shots).
        # And Total Team Seconds is 2 * (Real Time).
        # So ratios work out.
        league_norm_grid = league_grid_sum / total_team_seconds
        
        # Save Baseline
        np.save(os.path.join(out_root, 'baseline.npy'), league_norm_grid)
        
        # 2. Compute Rates and Percentiles
        # Prepare lists for percentile calculation
        all_xgf60 = []
        all_xga60 = []
        
        # Calculate rates first
        for tid, s in team_stats.items():
            if s['team_seconds'] > 0:
                s['team_xg_per60'] = (s['team_xgs'] / s['team_seconds']) * 3600
                s['other_xg_per60'] = (s['other_xgs'] / s['team_seconds']) * 3600
                all_xgf60.append(s['team_xg_per60'])
                all_xga60.append(s['other_xg_per60'])
            else:
                s['team_xg_per60'] = 0.0
                s['other_xg_per60'] = 0.0
                
        all_xgf60 = np.array(all_xgf60)
        all_xga60 = np.array(all_xga60)
        
        league_xgf60_mean = np.mean(all_xgf60) if len(all_xgf60) > 0 else 1.0
        league_xga60_mean = np.mean(all_xga60) if len(all_xga60) > 0 else 1.0
        
        summary_list = []
        
        # --- PHASE 3: PLOT ---
        for tid, grid in team_grids.items():
            tname = t_map.get(tid, str(tid))
            s = team_stats.get(tid)
            if not s or s['team_seconds'] <= 0: continue
            
            # Percentiles
            off_pct = percentileofscore(all_xgf60, s['team_xg_per60'])
            def_pct = 100 - percentileofscore(all_xga60, s['other_xg_per60']) # Lower GA is better (higher percentile rank usually means "better")
            
            # Relative Pct Change
            rel_off_pct = 100 * (s['team_xg_per60'] - league_xgf60_mean) / league_xgf60_mean if league_xgf60_mean > 0 else 0
            rel_def_pct = 100 * (s['other_xg_per60'] - league_xga60_mean) / league_xga60_mean if league_xga60_mean > 0 else 0
            
            # Shot shares
            tot_att = s['team_attempts'] + s['other_attempts']
            t_att_pct = 100 * s['team_attempts'] / tot_att if tot_att > 0 else 0.0
            o_att_pct = 100 * s['other_attempts'] / tot_att if tot_att > 0 else 0.0
            
            # Update Stats Dict for Plotter
            s['off_percentile'] = off_pct
            s['def_percentile'] = def_pct
            s['rel_off_pct'] = rel_off_pct
            s['rel_def_pct'] = rel_def_pct
            s['home_shot_pct'] = t_att_pct
            s['away_shot_pct'] = o_att_pct
            s['away_shot_pct'] = o_att_pct
            s['team_name'] = tname
            
            # Map Team/Other to Home/Away for Summary Text
            s['home_goals'] = s['team_goals']
            s['away_goals'] = s['other_goals']
            s['home_xg'] = s['team_xgs']
            s['away_xg'] = s['other_xgs']
            s['have_xg'] = True
            s['home_attempts'] = s['team_attempts']
            s['away_attempts'] = s['other_attempts']
            
            # Add to Summary List (for Scatter/CSV)
            summary_list.append({
                'team': tname,
                'team_xg_per60': s['team_xg_per60'],
                'other_xg_per60': s['other_xg_per60'],
                'gf_pct': 100 * s['team_goals'] / (s['team_goals'] + s['other_goals']) if (s['team_goals'] + s['other_goals']) > 0 else 0,
                'xgf_pct': 100 * s['team_xgs'] / (s['team_xgs'] + s['other_xgs']) if (s['team_xgs'] + s['other_xgs']) > 0 else 0
            })
            
            # 1. Raw Plot (Hybrid)
            try:
                if not df_cond.empty:
                    mask_team = (df_cond['team_id'] == tid) | (df_cond['home_id'] == tid) | (df_cond['away_id'] == tid)
                    df_team_events = df_cond[mask_team].copy()
                    
                    if not df_team_events.empty:
                        custom_styles = {
                            'goal': {'marker': 'D', 'size': 80, 'team_color': 'green', 'not_team_color': 'red', 'zorder': 10},
                            'shot-on-goal': {'marker': 'o', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'},
                            'missed-shot': {'marker': 'x', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'},
                            'blocked-shot': {'marker': '^', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'}
                        }
                        
                        raw_out = os.path.join(out_root, f"{tname}_season_summary.png")
                        plot_events(
                            df_team_events,
                            events_to_plot=['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'],
                            event_styles=custom_styles,
                            out_path=raw_out,
                            heatmap_split_mode='team_not_team',
                            team_for_heatmap=tid,
                            summary_stats=s,
                            title=f"{tname} Season Summary",
                            return_heatmaps=False
                        )
            except Exception as e:
                print(f"Failed raw plot for {tname}: {e}")
                
            # 2. Relative xG Map
            # Team Norm Grid = Team Sum Grid / Team Seconds
            team_norm_grid = grid / s['team_seconds']
            rel_grid = team_norm_grid - league_norm_grid
            
            # Blueline Masking
            # Standard rink: -100 to 100.
            # Blue lines at -25 and +25? No.
            # Blue lines are at +/- 25ft from CENTER red line.
            # So Neutral Zone is between -25 and +25.
            # Grid shape (86, 201). X is columns (201).
            # Map X coord (-100 to 100) to indices 0..200.
            # index = x + 100.
            # Mask indices 75 to 125?
            # Let's map coords.
            # x values: linspace(-100, 100, 201)
            # -25 maps to index 75.
            # +25 maps to index 125.
            
            # Mask logical: where abs(x) < 25.
            # rel_grid is (86, 201).
            mask = np.ones_like(rel_grid, dtype=bool)
            # Create indexing array for columns
            x_indices = np.arange(rel_grid.shape[1])
            # Coords: x = x_indices - 100
            xs = x_indices - 100
            mask[:, np.abs(xs) < 25] = False
            
            rel_grid_masked = np.where(mask, rel_grid, 0) # Set neutral zone to 0 (white/transparent equivalent data)
            # Or use masked array for transparency
            rel_grid_masked_ma = np.ma.masked_where(~mask, rel_grid)
            
            out_path = os.path.join(out_root, f"{tname}_relative.png")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                draw_rink(ax=ax)
                
                extent = (-100, 100, -42.5, 42.5)
                cmap = plt.get_cmap('RdBu_r')
                
                # Determine Vmax from masked data (ignore the 0s we just set if we want dynamic range)
                vmax = np.max(np.abs(rel_grid_masked_ma))
                if vmax == 0: vmax = 1.0
                norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
                
                # Plot
                # Use rel_grid_masked_ma to have transparency in neutral zone if supported by imshow, 
                # or just plot zeros as white (0 is white in RdBu_r usually).
                # Transparency is better.
                im = ax.imshow(rel_grid_masked_ma, extent=extent, origin='lower', cmap=cmap, norm=norm, alpha=0.8)
                
                # Summary Text
                display_cond = f"{cond} | Empty Net: False"
                add_summary_text(ax=ax, stats=s, main_title=f"{tname} Relative xG", 
                                 is_season_summary=True, team_name=tname, full_team_name=tname, filter_str=display_cond)
                                 
                ax.axis('off')
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
                cbar.set_label('Excess xG per 60', rotation=270, labelpad=15)
                
                fig.savefig(out_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting relative {tname}: {e}")
                
        # 3. Scatter Plot
        # Save JSON/CSV first
        if summary_list:
            df_sum = pd.DataFrame(summary_list)
            df_sum.to_csv(os.path.join(out_root, 'team_summary.csv'), index=False)
            with open(os.path.join(out_root, 'team_summary.json'), 'w') as f:
                json.dump(summary_list, f)
                
            try:
                generate_scatter_plot(summary_list, out_root, cond)
            except Exception as e:
                print(f"Failed to generate scatter plot: {e}")
        
        print(f"  Processed {len(summary_list)} teams for {cond}.")

if __name__ == '__main__':
    run_league_analysis()
