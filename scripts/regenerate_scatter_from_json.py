import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from puck.analyze import generate_scatter_plot
except ImportError:
    # Fallback if function not exposed directly or name differs
    print("Could not import generate_scatter_plot from puck.analyze. Checking local usage.")
    pass

def main():
    season = "20252026"
    out_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
    json_path = out_dir / "team_stats_summary.json"
    
    if not json_path.exists():
        print("JSON stats not found.")
        return

    with open(json_path, 'r') as f:
        grand_stats = json.load(f)
        
    print(f"Loaded stats for {len(grand_stats)} teams.")
    
    output_states = ['5v5', '5v4', '4v5']
    
    for plot_state in output_states:
        print(f"Generating Scatter for {plot_state}...")
        
        # Rebuild summary_list
        summary_list = []
        all_xgf60 = []
        all_xga60 = []
        
        # 1. Collect Rates for Averages
        for team, buckets in grand_stats.items():
            if plot_state not in buckets: continue
            s = buckets[plot_state]
            sec = s['seconds']
            if sec > 0:
                xgf60 = (s['xg_for'] / sec) * 3600
                xga60 = (s['xg_against'] / sec) * 3600
                all_xgf60.append(xgf60)
                all_xga60.append(xga60)
                
        avg_xgf60 = np.mean(all_xgf60) if all_xgf60 else 0
        avg_xga60 = np.mean(all_xga60) if all_xga60 else 0
        
        # 2. Build Dicts
        for team, buckets in grand_stats.items():
            if plot_state not in buckets: continue
            s = buckets[plot_state]
            sec = s['seconds']
            if sec == 0: continue
            
            xgf60 = (s['xg_for'] / sec) * 3600
            xga60 = (s['xg_against'] / sec) * 3600
            
            stats_dict = s.copy()
            stats_dict['team'] = team
            stats_dict['team_xg_per60'] = xgf60
            stats_dict['other_xg_per60'] = xga60
            
            # Percents (Approximate for visual)
            stats_dict['rel_off_pct'] = 100 * (xgf60 - avg_xgf60) / avg_xgf60 if avg_xgf60 else 0
            stats_dict['rel_def_pct'] = 100 * (xga60 - avg_xga60) / avg_xga60 if avg_xga60 else 0
            
            # Shot pct
            tot_att = s['attempts_for'] + s['attempts_against']
            stats_dict['home_shot_pct'] = 100 * s['attempts_for'] / tot_att if tot_att else 0
            
            summary_list.append(stats_dict)
            
        # 3. Plot
        try:
            # We call the imported function
            # Need to verify signature instructions: 
            # generate_scatter_plot(summary_list, str(out_dir), condition_name=f"Mixed Effects {plot_state}")
            
            generate_scatter_plot(summary_list, str(out_dir), condition_name=f"Mixed Effects {plot_state}")
            
            # Manual Rename Fix
            default_scatter = out_dir / "scatter.png"
            if default_scatter.exists():
                new_scatter = out_dir / f"scatter_{plot_state}.png"
                if new_scatter.exists():
                     new_scatter.unlink()
                default_scatter.rename(new_scatter)
                print(f"Saved {new_scatter}")
            else:
                print(f"Warning: scatter.png not found for {plot_state}")
                
        except Exception as e:
            print(f"Error plotting {plot_state}: {e}")

if __name__ == "__main__":
    main()
