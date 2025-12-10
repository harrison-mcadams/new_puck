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
from puck.plot import add_summary_text
from puck.analyze import compute_relative_map

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

    t1 = df_meta[['home_id', 'home_abb']].rename(columns={'home_id': 'id', 'home_abb': 'abb'})
    t2 = df_meta[['away_id', 'away_abb']].rename(columns={'away_id': 'id', 'away_abb': 'abb'})
    t_map = pd.concat([t1, t2]).drop_duplicates('id').set_index('id')['abb'].to_dict()
    del df_meta
    gc.collect()

    conditions = ['5v5', '5v4', '4v5']
    
    for cond in conditions:
        print(f"\nProcessing Condition: {cond}")
        out_root = os.path.join('analysis', 'league', season, cond)
        os.makedirs(out_root, exist_ok=True)
        
        # Scan caches
        files =  sorted([f for f in os.listdir(cache_dir) if f.endswith(f'_{cond}.npz')])
        print(f"  Found {len(files)} game files.")
        
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
                            
                            grid = data[k]
                            
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
                             # Wait, league_seconds should be sum of ALL intervals in league.
                             # If we sum team_seconds for all teams, we get 2x total time (home+away).
                             # So league_seconds = Sum(TeamSecs) / 2
                             
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
                'xg_for_60': xg_f60,
                'xg_against_60': xg_a60,
                'gf_pct': 100 * stats['team_goals'] / (stats['team_goals'] + stats['other_goals']) if (stats['team_goals'] + stats['other_goals']) > 0 else 0,
                'xgf_pct': 100 * xg_f / (xg_f + xg_a) if (xg_f + xg_a) > 0 else 0
            })
            
            # Plot Relative
            # Team - League
            # Note: compute_relative_map expects 'league_map' to be the baseline rate
            # We calculated league_norm correctly above.
            
            # Relative Plot
            out_path = os.path.join(out_root, f"{tname}.png")
            
            try:
                # We use other_map=None for Team vs League? 
                # No, standard relative map is Team For vs League For.
                # analyze.season uses:
                # combined_rel_map = team_norm - league_norm (simplification) or log ratio.
                
                # Let's use compute_relative_map but we only have 1 map (Team For).
                # Wait, compute_relative_map takes (team_map, league_map, ...).
                # It handles the math.
                
                # Dummy 'other' map
                # Actually compute_relative_map is for Player vs Team OR Team vs League.
                # If we pass team_map and league_map, it works.
                
                combined_rel, rel_off_pct, rel_def_pct, _, _ = compute_relative_map(
                    team_norm * sec, league_norm * sec, sec, 
                    None, sec, # No 'other' comparison for pure team map?
                    # Wait, analyze.season passes league_res as 'league_data'.
                    # And calls xgs_map.
                    # xgs_map plots the absolute map.
                    # Then it plots relative.
                    # We need both.
                )
                
                # Actually analyze.season mostly plots xG For vs xG Against if cond is 5v5?
                # Or just Team Offense?
                # Usually it plots Team Offense Relative to League Average.
                
                fig, ax = plt.subplots(figsize=(10, 5))
                draw_rink(ax=ax)
                from matplotlib.colors import SymLogNorm
                norm = SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=-0.0006, vmax=0.0006, base=10)
                extent = (-100.5, 100.5, -42.5, 42.5) 
                
                cmap = plt.get_cmap('RdBu_r')
                try: cmap.set_bad(color=(1,1,1,0)) 
                except: pass
                
                # Calc relative manually if compute_relative_map is complex with None
                # Rel = TeamRate - LeagueRate
                rel_grid = team_norm - league_norm
                
                m = np.ma.masked_invalid(rel_grid)
                im = ax.imshow(m, extent=extent, origin='lower', cmap=cmap, norm=norm)
                
                txt_props = {
                    'home_xg': xg_f, 'away_xg': xg_a, 'have_xg': True,
                    'home_goals': stats['team_goals'], 'away_goals': stats['other_goals'],
                    'home_attempts': stats['team_attempts'], 'away_attempts': stats['other_attempts'],
                }
                
                add_summary_text(ax=ax, stats=txt_props, main_title=tname, is_season_summary=True,
                                 team_name=tname, full_team_name=tname, filter_str=cond)
                
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                fig.savefig(out_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                # print(f"Error plotting {tname}: {e}")
                pass
                
        # Save Summary
        df_sum = pd.DataFrame(summary_list)
        df_sum.to_csv(os.path.join(out_root, 'team_summary.csv'), index=False)
        with open(os.path.join(out_root, 'team_summary.json'), 'w') as f:
            json.dump(summary_list, f)
            
        print(f"  Generated stats/plots for {len(summary_list)} teams.")

if __name__ == '__main__':
    run_league_analysis()
