"""
generate_mixed_heatmaps.py

Generates Heatmaps (Absolute & Relative) for all teams in 2025-2026
using the Mixed Effects xG Model.
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

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects, fit_nested_xgs, fit_xgs
from puck.plot import plot_relative_map, plot_events
from puck.rink import draw_rink
from puck import config

def main():
    season = "20252026"
    print(f"--- Generating Mixed Effects Heatmaps for {season} ---")

    # 1. Load 2025-2026 Data
    print("Loading Data...")
    # Fix: Check data/20252026.csv (root of data/ dir) first
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

    # 2. Train/Fit Mixed Effects Model
    print("Fitting Mixed Effects Model...")
    # We fit on the whole season dataset to get the descriptive model for this season
    me_model = mixed_effects.MixedEffectsXG(
        n_estimators=100, 
        l2_reg=1.0, 
        learning_rate=0.5,
        group_col='team_name'
    )
    me_model.fit(df)
    
    # 3. Predict & Overwrite xGs
    print("Predicting Mixed xG...")
    probs = me_model.predict_proba(df)[:, 1]
    df['xgs'] = probs
    
    # 4. Aggregation Logic
    # We need to build grids.
    # Grid Specs: x in [-100, 100], y in [-42.5, 42.5]
    # Standard puck logic uses 200 bins for x (1 ft) and 85 bins for y (1 ft)?
    # Let's verify commonly used shape. usually (85, 200) or (170, 200). 
    # analyze.py uses standard histogram2d. 
    # Let's check ranges.
    
    BIN_X = np.linspace(-100, 100, 201) # 200 bins
    BIN_Y = np.linspace(-42.5, 42.5, 86) # 85 bins
    # Shape will be (200, 85) typically, but numpy histogram2d returns (nx, ny).
    # NOTE: plot_relative_map expects (Y, X) usually? Or (X, Y)?
    # plot_relative_map -> imshow using calculations.
    # Let's stick to (85, 200) for "image" logic (Rows=Y, Cols=X).
    # So histogram2d(y, x, bins=[BIN_Y, BIN_X])
    
    league_grid_sum = np.zeros((85, 200))
    league_seconds = 0.0
    
    team_stats = {} # team -> {grid_for, grid_against, xgf, xga, goals, opp_goals, seconds, attempts, opp_attempts}
    
    teams = sorted(df['team_name'].unique())
    print(f"Aggregating stats for {len(teams)} teams...")
    
    # Pre-calculate timing for games
    # We need Time on Ice per Team.
    # Simple Approx: 5v5 duration per game?
    # Better: Use the dataframe's 'period_time' diffs? 
    # Assuming standard 60 mins per game? No, we filter for 5v5 usually.
    # Let's Filter for 5v5 FIRST as per standard relative maps
    mask_5v5 = (df['game_state'] == '5v5') & (df['is_net_empty'] == 0)
    df_5v5 = df[mask_5v5].copy()
    game_ids = df_5v5['game_id'].unique()
    
    # Parallel Processing using Joblib
    from joblib import Parallel, delayed
    print(f"Processing {len(game_ids)} games in PARALLEL...")

    def process_game(gid):
        try:
            g_df = df_5v5[df_5v5['game_id'] == gid]
            if g_df.empty: return None
            
            # Determine teams
            home_team = g_df['home_abb'].iloc[0]
            away_team = g_df['away_abb'].iloc[0]
            
            # Timing
            try:
                # We need to import timing inside worker if not available? 
                # joblib usually handles imports if process spawned
                from puck import timing
                intervals = timing.get_game_intervals_cached(gid, season, {'game_state': ['5v5']})
                seconds = sum(e-s for s,e in intervals)
            except:
                seconds = 2800.0 
            
            game_res = {} # team -> stats
            
            for team, opp in [(home_team, away_team), (away_team, home_team)]:
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
                    
                game_res[team] = stats
                
            return game_res
        except Exception as e:
            return None

    results = Parallel(n_jobs=-1, verbose=1)(delayed(process_game)(gid) for gid in game_ids)
    
    # Merge Results
    print("Merging results...")
    for res in results:
        if not res: continue
        
        # League Seconds (Add once per game or sum per team?)
        # Logic matches previous loop: league_seconds += (seconds * 2)
        # Here we just iterate teams.
        # Wait, previous loop: league_seconds += (seconds * 2)
        # Here: each team dict has 'seconds'.
        # So we can just sum team seconds later?
        # Yes, team_stats[t]['seconds'] accumulated.
        
        for team, stats in res.items():
            if team not in team_stats:
                team_stats[team] = {
                    'grid_for': np.zeros((85, 200)),
                    'grid_against': np.zeros((85, 200)),
                    'seconds': 0.0, 'xg_for': 0.0, 'xg_against': 0.0,
                    'goals_for': 0, 'goals_against': 0,
                    'attempts_for': 0, 'attempts_against': 0
                }
            ts = team_stats[team]
            ts['seconds'] += stats['seconds']
            ts['xg_for'] += stats['xg_for']
            ts['xg_against'] += stats['xg_against']
            ts['goals_for'] += stats['goals_for']
            ts['goals_against'] += stats['goals_against']
            ts['attempts_for'] += stats['attempts_for']
            ts['attempts_against'] += stats['attempts_against']
            ts['grid_for'] += stats['grid_for']
            ts['grid_against'] += stats['grid_against']
            
    # Compute League Seconds
    league_seconds = sum(ts['seconds'] for ts in team_stats.values())
                    
    # League Sum
    for t in team_stats:
        league_grid_sum += team_stats[t]['grid_for']
        league_grid_sum += team_stats[t]['grid_against']
        
    league_norm_grid = league_grid_sum / league_seconds
    
    # 5. Stats & Percentiles
    all_xgf60 = []
    all_xga60 = []
    
    for t, s in team_stats.items():
        if s['seconds'] > 0:
            s['xgf60'] = (s['xg_for'] / s['seconds']) * 3600
            s['xga60'] = (s['xg_against'] / s['seconds']) * 3600
        else:
            s['xgf60'] = 0; s['xga60'] = 0
        all_xgf60.append(s['xgf60'])
        all_xga60.append(s['xga60'])
        
    avg_xgf60 = np.mean(all_xgf60)
    avg_xga60 = np.mean(all_xga60)
    
    # 6. Plotting
    out_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t, s in team_stats.items():
        if s['seconds'] < 100: continue
        
        # 6a. Compute Relative Grid
        # Team Norm
        grid_total = s['grid_for'] + s['grid_against']
        team_norm = grid_total / s['seconds']
        
        # Relative (diff per 100 sq ft per 60 min)
        # Apply Gaussian Smoothing (Sigma=2 -> ~2 ft) to smooth plain histogram
        # Note: We smooth League Norm too? Usually smoother relative = smooth(team) - smooth(league)
        # Let's smooth the final diff instead?
        # Standard: smooth(data) then subtract.
        # But (smooth(A) - smooth(B)) == smooth(A-B) due to linearity of convolution.
        
        # We smooth the Team Norm
        team_norm_smooth = gaussian_filter(team_norm, sigma=2.0)
        # We assume league_norm_grid needs smoothing too if it wasn't already?
        # It's constructed from histogram so it's blocky.
        league_norm_smooth = gaussian_filter(league_norm_grid, sigma=2.0)
        
        rel_grid = (team_norm_smooth - league_norm_smooth) * 3600 * 100
        
        # 6b. Stats Dictionary for 'add_summary_text'
        stats_dict = s.copy()
        
        # Percentiles
        stats_dict['off_percentile'] = percentileofscore(all_xgf60, s['xgf60'])
        stats_dict['def_percentile'] = 100 - percentileofscore(all_xga60, s['xga60'])
        
        # Relative %
        stats_dict['rel_off_pct'] = 100 * (s['xgf60'] - avg_xgf60) / avg_xgf60 if avg_xgf60 else 0
        stats_dict['rel_def_pct'] = 100 * (s['xga60'] - avg_xga60) / avg_xga60 if avg_xga60 else 0
        
        # Shot Share
        tot_att = s['attempts_for'] + s['attempts_against']
        stats_dict['home_shot_pct'] = 100 * s['attempts_for'] / tot_att if tot_att else 0
        stats_dict['away_shot_pct'] = 100 * s['attempts_against'] / tot_att if tot_att else 0
        
        # fix: Map xG/60 to expected keys for add_summary_text
        stats_dict['team_xg_per60'] = s['xgf60']
        stats_dict['other_xg_per60'] = s['xga60']
        
        # Keys expected by add_summary_text
        stats_dict['home_goals'] = s['goals_for']
        stats_dict['away_goals'] = s['goals_against']
        stats_dict['home_xg'] = s['xg_for']
        stats_dict['away_xg'] = s['xg_against']
        stats_dict['have_xg'] = True
        
        # 6c. Plot Absolute (Density)
        # Actually user asked for "heatmap of xgs for and against" -> Absolute
        # We can implement a simple density plot or reuse plot_events with `return_heatmaps=False`?
        # plot_events generates a scatter/density hybrid.
        # Let's make the Relative one first as it matches `run_league_stats`.
        
        # PLOT RELATIVE
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = plot_relative_map(
            ax=ax,
            rel_grid=rel_grid,
            title=f"{t} Mixed Model Relative xG (2025-26)",
            stats=stats_dict,
            team_name=t,
            full_team_name=t,
            cond="5v5",
            mask_neutral_zone=True
        )
        
        # Standard Decor
        fig.patch.set_facecolor('white')
        
        # Colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        out_rel = out_dir / f"{t}_mixed_relative.png"
        fig.savefig(out_rel, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # PLOT ABSOLUTE (Raw xG Density)
        # We can use our grids directly
        fig_abs, ax_abs = plt.subplots(figsize=(10,6))
        draw_rink(ax_abs)
        
        # Mask
        mask_x = np.abs(np.linspace(-100, 100, 200)) < 25
        mask = np.tile(mask_x, (85, 1))
        
        abs_grid_view = team_norm_smooth * 3600 * 100 # Per 60 per 100 sq ft
        abs_grid_ma = np.ma.masked_where(mask, abs_grid_view)
        
        # Plot
        # Use simple hot colormap
        ax_abs.imshow(abs_grid_ma, extent=[-100, 100, -42.5, 42.5], origin='lower', cmap='inferno')
        ax_abs.set_title(f"{t} Mixed xG Density (5v5)")
        
        # Reuse summary text?
        # Absolute plots usually have simple stats? 
        # User said "add the exact same summary text as those summary figures produced by daily.py"
        # daily.py produces Relative ones with text used above.
        # It also produces "Season Summary" (Raw Scatter).
        # Let's add the SAME summary text to this absolute heatmap too.
        from puck.plot import add_summary_text
        add_summary_text(ax_abs, stats_dict, f"{t} Mixed xG Density", is_season_summary=True, team_name=t)
        
        out_abs = out_dir / f"{t}_mixed_absolute.png"
        fig_abs.savefig(out_abs, dpi=120, bbox_inches='tight')
        plt.close(fig_abs)
        
    print(f"Done. Maps saved to {out_dir}")

if __name__ == "__main__":
    main()
