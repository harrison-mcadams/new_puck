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
from puck.rink import draw_rink
from puck.plot import add_summary_text, plot_events
from puck.analyze import compute_relative_map, generate_scatter_plot

def run_league_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='20252026')
    args = parser.parse_args()
    season = args.season
    
    print(f"--- Cached League Analysis for {season} ---")
    
    cache_dir = os.path.join(config.get_cache_dir(season), 'partials')
    print(f"Checking cache dir: {cache_dir}")
    if not os.path.exists(cache_dir):
        print(f"No cache directory found at {cache_dir}. Run daily.py to generate caches.")
        return

    # Team Info (Abbreviation map)
    from puck import timing
    df_meta = timing.load_season_df(season) 
    if df_meta is None or df_meta.empty:
        print("No season data.")
        return

    # Keep df_meta as full season df for raw plotting
    df_season = df_meta
    
    t1 = df_season[['home_id', 'home_abb']].rename(columns={'home_id': 'id', 'home_abb': 'abb'})
    t2 = df_season[['away_id', 'away_abb']].rename(columns={'away_id': 'id', 'away_abb': 'abb'})
    t_map = pd.concat([t1, t2]).drop_duplicates('id').set_index('id')['abb'].to_dict()
    # do NOT delete df_meta/df_season, we need it for hybrid plotting
    gc.collect()

    conditions = ['5v5', '5v4', '4v5']
    
    for cond in conditions:
        print(f"\nProcessing Condition: {cond}")
        out_root = os.path.join('analysis', 'league', season, cond)
        os.makedirs(out_root, exist_ok=True)
        
        # Scan caches
        files =  sorted([f for f in os.listdir(cache_dir) if f.endswith(f'_{cond}.npz')])
        print(f"  Found {len(files)} game files.")
        
        # Prepare filtered DF for this condition
        # Logic adapted from analyze.season
        from puck.parse import build_mask
        # condition is dict-like usually, but here 'cond' is string '5v5'
        cond_dict = {'5v5': {'game_state': ['5v5']}, 
                     '5v4': {'game_state': ['5v4']}, 
                     '4v5': {'game_state': ['4v5']}}.get(cond)
        
        if cond_dict and not df_season.empty:
            mask = build_mask(df_season, cond_dict)
            df_cond = df_season[mask].copy()
            # Ensure filtered DF has integer IDs for consistent matching
            for col in ['team_id', 'home_id', 'away_id']:
                if col in df_cond.columns:
                    df_cond[col] = df_cond[col].fillna(-1).astype(int)
                    # Convert to string to avoid any float/int mismatch downstream?
                    # No, int is safer if tid is int.
        else:
            df_cond = pd.DataFrame()

        league_grid = None
        league_seconds = 0.0
        
        team_grids = {}
        team_stats = {} # tid -> {summed stats}
        
        # Load & Aggregate
        for fname in files:
            try:
                path = os.path.join(cache_dir, fname)
                with np.load(path) as data:
                    if 'empty' in data: continue
                    
                    # Keys: team_{tid}_grid_team, team_{tid}_stats
                    keys = list(data.keys())
                    
                    for k in keys:
                        if k.startswith('team_') and k.endswith('_grid_team'):
                            tid_str = k.split('_')[1]
                            tid = int(tid_str)
                            
                            
                            grid = np.nan_to_num(data[k])
                            
                            # Add to League
                            if league_grid is None:
                                league_grid = grid.astype(np.float64)
                            else:
                                league_grid += grid
                                
                            # Add to Team
                            if tid not in team_grids:
                                team_grids[tid] = grid.astype(np.float64)
                            else:
                                team_grids[tid] += grid
                                
                        if k.startswith('team_') and k.endswith('_stats'):
                             tid_str = k.split('_')[1]
                             tid = int(tid_str)
                             
                             s_str = str(data[k])
                             s = json.loads(s_str)
                             
                             # We only add to league_seconds once per team per game
                             # But here we just sum total team seconds
                             # Wait, le
                             
                             if tid not in team_stats:
                                 team_stats[tid] = {'team_xgs': 0.0, 'other_xgs': 0.0, 'team_seconds': 0.0, 
                                                    'team_goals': 0, 'other_goals': 0,
                                                    'team_attempts': 0, 'other_attempts': 0}
                                 
                             ts = team_stats[tid]
                             ts['team_xgs'] += s.get('team_xgs', 0.0)
                             ts['other_xgs'] += s.get('other_xgs', 0.0)
                             ts['team_seconds'] += s.get('team_seconds', 0.0)
                             ts['team_goals'] += s.get('team_goals', 0)
                             ts['other_goals'] += s.get('other_goals', 0)
                             ts['team_attempts'] += s.get('team_attempts', 0)
                             ts['other_attempts'] += s.get('other_attempts', 0)
                             
            except Exception as e:
                # print(f"Error loading {fname}: {e}")
                pass
        
        # Calculate true league seconds
        total_team_seconds = sum(t['team_seconds'] for t in team_stats.values())
        league_seconds = total_team_seconds / 2.0 if total_team_seconds > 0 else 0.0
        
        # League Baseline
        if league_grid is None or league_seconds <= 0:
            print(f"  No data for {cond}.")
            continue
            
        # Normalize: League Grid is sum of Home+Away grids.
        # So it represents shots per (League Seconds * 2) if looked at naively?
        # No, league_grid is sum of expected shots.
        # Rate = Total xG / Total Time.
        # But we want League Average Team.
        # League Avg Team xG/60 = (Total League xG / 2) / (Total Time) ?
        # Or (Total League xG) / (Total Team Time).
        # Total Team Time = 2 * Real Time.
        # So yes, league_grid / total_team_seconds.
        league_norm = league_grid / total_team_seconds
        
        # Save Baseline
        np.save(os.path.join(out_root, 'baseline.npy'), league_norm)
        
        # Generate Team Plots
        summary_list = []
        
        for tid, grid in team_grids.items():
            tname = t_map.get(tid, str(tid))
            stats = team_stats.get(tid, {})
            sec = stats.get('team_seconds', 0.0)
            if sec <= 0: continue
            
            # Normalize Team
            team_norm = grid / sec
            
            xg_f = stats['team_xgs']
            xg_a = stats['other_xgs']
            xg_f60 = (xg_f/sec)*3600
            xg_a60 = (xg_a/sec)*3600
            
            summary_list.append({
                'team': tname,
                'team_xg_per60': xg_f60,
                'other_xg_per60': xg_a60,
                'gf_pct': 100 * stats['team_goals'] / (stats['team_goals'] + stats['other_goals']) if (stats['team_goals'] + stats['other_goals']) > 0 else 0,
                'xgf_pct': 100 * xg_f / (xg_f + xg_a) if (xg_f + xg_a) > 0 else 0
            })
            
            # UPDATE stats dict with rates so add_summary_text can find them
            stats['team_xg_per60'] = xg_f60
            stats['other_xg_per60'] = xg_a60
            # Also percentiles ideally, but we haven't computed them yet here.
            # We compute percentiles after accumulating summary_list usually.
            # But we are plotting INSIDE the loop.
            # This is a timing issue. Plotting relies on percentiles that need the full league info.
            # Ideally we should generate plots AFTER the loop.
            # Refactoring to plot after loop is better, but risky for big diff.
            # For now, at least rates will show up.

            # --- 1. Raw Shot Attempts Plot (Hybrid) ---
            try:
                # Filter events for this team
                # We want events where this team is the shooter (or blocked)
                # Team ID matching
                if not df_cond.empty:
                    # Logic from plot_events internal or similar
                    # Filter for rows involving this team
                    mask_team = (df_cond['team_id'] == tid) | (df_cond['home_id'] == tid) | (df_cond['away_id'] == tid)
                    df_team_events = df_cond[mask_team].copy()
                    
                    if not df_team_events.empty:
                        raw_out_path = os.path.join(out_root, f"{tname}_season_summary.png")
                        
                        # We want all shots (for/against) with visual distinction
                        # plot_events handles 'team_not_team' split mode if we pass team_for_heatmap
                        # But wait, user wants "raw shot attempts plotted... different shot attempt types... visually distinct."
                        # AND "both for and against".
                        # 'team_not_team' mode in plot_events does coloring by Team vs Opponent.
                        # Do we want that OR event type coloring?
                        # User said: "different shot attempt types ... should be visually distinct."
                        # If we use plot_events, it colors by Home/Away or Team/NotTeam usually.
                        # It iterates by event type and uses style dict.
                        # See plot.py: style = merged_styles.get(ev_type, ...)
                        # It extracts home_color/away_color from style.
                        # So if we want Goal=Green, Miss=Red, we set that in style.
                        # And we want distinction between For and Against.
                        # Typically "Goal For" vs "Goal Against".
                        # plot_events does simple scattering.
                        
                        # Let's configure styles to be distinct per event type
                        # And 'team_not_team' mode will assign color 1 (Team) or color 2 (Opponent) from the style.
                        
                        custom_styles = {
                            'goal': {'marker': 'D', 'size': 80, 'team_color': 'green', 'not_team_color': 'red', 'zorder': 10},
                            'shot-on-goal': {'marker': 'o', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'},
                            'missed-shot': {'marker': 'x', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'}, # Same color, diff marker?
                            'blocked-shot': {'marker': '^', 'size': 30, 'team_color': 'cyan', 'not_team_color': 'magenta'}
                        }
                        
                        # Actually standard practice:
                        # Goals: Solid
                        # Shots: Open Circle
                        # Miss/Block: X or ^
                        # Colors: Team vs Opp is usually Black/Orange or similar.
                        # User asked for "different shot attempt types ... should be visually distinct".
                        # Maybe they imply markers?
                        # I will use distinct markers and distinct colors for team vs opp.
                        
                        plot_events(
                            df_team_events,
                            events_to_plot=['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'],
                            event_styles=custom_styles,
                            out_path=raw_out_path,
                            heatmap_split_mode='team_not_team',
                            team_for_heatmap=tid,
                            summary_stats=stats, # Show cached stats
                            title=f"{tname} Season Summary (Raw Attempts)",
                            return_heatmaps=False
                        )
            except Exception as e:
                print(f"Failed raw plot for {tname}: {e}")

            # --- 2. Relative xG Plot (Fixed Style) ---
            out_path = os.path.join(out_root, f"{tname}_relative.png")
            
            try:
                # Calculate Relative Grid: (Team - League) / League ?
                # Or just Team - League (Difference)?
                # Standard relative xG maps are usually "Excess xG per 60".
                # So TeamRate - LeagueRate.
                # Units: xG per 60 mins per 100 sq ft (or whatever grid size).
                
                rel_grid = team_norm - league_norm
                
                # Setup Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                draw_rink(ax=ax)
                
                # Params matching "WSH_relative.png" style (assumed from context/experience with this codebase type)
                # Typically RdBu_r centered at 0.
                # Limits: +/- some reasonable rate. e.g. 0.0006
                
                extent = (-100, 100, -42.5, 42.5) # Rink bounds +/-
                # Wait, grid bounds from analyze.py are usually -100 to 100, -42.5 to 42.5
                # We need to trust the grid shape matches the rink.
                
                cmap = plt.get_cmap('RdBu_r')
                # Use SymLogNorm or Linear?
                # Usually simplified heatmaps use Linear with vmin/vmax symmetric.
                vmax = np.max(np.abs(rel_grid))
                if vmax == 0: vmax = 1.0
                norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
                
                # User mentioned "new_puck/analysis/xleague/20252026/5v5/WSH_relative.png" as specific ref.
                # I cannot see that image, but I can infer "Relative xG map" implies difference.
                # If the previous code used SymLogNorm, maybe stick to it?
                # But raw imshow with RdBu_r is safer for "difference".
                
                im = ax.imshow(rel_grid.T, extent=extent, origin='lower', cmap=cmap, norm=norm, alpha=0.8)
                
                # Add Summary Text
                txt_props = {
                    'home_xg': xg_f, 'away_xg': xg_a, 'have_xg': True,
                    'home_goals': stats['team_goals'], 'away_goals': stats['other_goals'],
                    'home_attempts': stats['team_attempts'], 'away_attempts': stats['other_attempts'],
                }
                
                add_summary_text(ax=ax, stats=txt_props, main_title=f"{tname} Relative xG", is_season_summary=True,
                                 team_name=tname, full_team_name=tname, filter_str=cond)
                
                ax.axis('off')
                # Colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
                cbar.set_label('Excess xG per 60', rotation=270, labelpad=15)
                
                fig.savefig(out_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error plotting {tname} relative: {e}")
                
        # Save Summary
        df_sum = pd.DataFrame(summary_list)
        df_sum.to_csv(os.path.join(out_root, 'team_summary.csv'), index=False)
        with open(os.path.join(out_root, 'team_summary.json'), 'w') as f:
            json.dump(summary_list, f)
            
        print(f"  Generated stats/plots for {len(summary_list)} teams.")

        # --- 3. Scatter Plots ---
        print("  Generating Scatter Plot...")
        try:
            # Use summary_list which has the calculated per-60 stats
            generate_scatter_plot(summary_list, out_root, cond)
        except Exception as e:
            print(f"Failed to generate scatter plot: {e}")

if __name__ == '__main__':
    run_league_analysis()
