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

    # Grid Specs
    BIN_X = np.linspace(-100, 100, 201) 
    BIN_Y = np.linspace(-42.5, 42.5, 86)
    
    # GRAND STATS BUCKET
    # Structure: grand_stats[team][condition]
    # conditions: '5v5', '5v4' (Advantage), '4v5' (Disadvantage)
    grand_stats = {} 
    
    processed_states = ['5v5', '5v4', '4v5'] 
    
    for state in processed_states:
        print(f"\n=== Processing Mixed Effects Model and Events for Global State: {state} ===")
        
        # Filter for current state
        mask_state = (df['game_state'] == state) & (df['is_net_empty'] == 0)
        df_state = df[mask_state].copy()
        
        if df_state.empty:
            print(f"No data for {state}, skipping.")
            continue
            
        print(f"  Data for {state}: {len(df_state)} rows")

        # 2. Train/Fit Mixed Effects Model (State Specific)
        print(f"  Fitting Mixed Effects Model for {state}...")
        me_model = mixed_effects.MixedEffectsXG(
            n_estimators=100, 
            l2_reg=1.0, 
            learning_rate=0.5,
            group_col='team_name'
        )
        me_model.fit(df_state)
        
        # 3. Predict & Overwrite xGs
        print(f"  Predicting Mixed xG for {state}...")
        probs = me_model.predict_proba(df_state)[:, 1]
        df_state['xgs'] = probs
        
        # 5. Process Games (Parallel)
        game_ids = df_state['game_id'].unique()
        print(f"  Processing {len(game_ids)} games...")

        def process_game(gid):
            try:
                g_df = df_state[df_state['game_id'] == gid]
                if g_df.empty: return None
                
                # Determine teams
                home_team = g_df['home_abb'].iloc[0]
                away_team = g_df['away_abb'].iloc[0]
                
                # Timing
                try:
                    from puck import timing
                    intervals = timing.get_game_intervals_cached(gid, season, {'game_state': [state]})
                    seconds = sum(e-s for s,e in intervals)
                except:
                    seconds = 2800.0 
                
                res = {
                    'game_id': gid,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_stats': None,
                    'away_stats': None
                }
                
                for role, team, opp in [('home', home_team, away_team), ('away', away_team, home_team)]:
                    events = g_df[g_df['team_name'] == team]
                    opp_events = g_df[g_df['team_name'] == opp]
                    
                    stats = {
                        'seconds': seconds,
                        'xg_for': events['xgs'].sum(),
                        'xg_against': opp_events['xgs'].sum(),
                        'goals_for': (events['event'] == 'goal').sum(),
                        'goals_against': (opp_events['event'] == 'goal').sum(),
                        'attempts_for': len(events),
                        'attempts_against': len(opp_events),
                        'grid_for': np.zeros((85, 200)),
                        'grid_against': np.zeros((85, 200))
                    }
                    
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

        results = Parallel(n_jobs=-1, verbose=1)(delayed(process_game)(gid) for gid in game_ids)
    
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

    print(f"Done. Maps saved to {out_dir}")

if __name__ == "__main__":
    main()
