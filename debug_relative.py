
import analyze
import pandas as pd
import numpy as np
import os

def debug_relative():
    season = '20252026'
    team = 'PHI'
    condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': team}
    
    print(f"Running debug for {team} {season} condition={condition}")
    
    # Call xgs_map directly
    out_path, ret_heat, ret_df, summary_stats = analyze.xgs_map(
        season=season,
        condition=condition,
        out_path='debug_map.png',
        return_heatmaps=True,
        return_filtered_df=True,
        show=False
    )
    
    print("Summary Stats:", summary_stats)
    
    if ret_heat:
        team_map = ret_heat.get('team')
        other_map = ret_heat.get('other')
        
        if team_map is not None:
            print(f"Team Map: shape={team_map.shape}, sum={np.nansum(team_map)}, max={np.nanmax(team_map)}")
        else:
            print("Team Map is None")
            
        if other_map is not None:
            print(f"Other Map: shape={other_map.shape}, sum={np.nansum(other_map)}, max={np.nanmax(other_map)}")
            
            # Check distribution (Left vs Right)
            # Left is indices 0-99 (approx), Right is 100-200
            mid = other_map.shape[1] // 2
            left_sum = np.nansum(other_map[:, :mid])
            right_sum = np.nansum(other_map[:, mid:])
            print(f"Other Map Left Sum: {left_sum}")
            print(f"Other Map Right Sum: {right_sum}")
        else:
            print("Other Map is None")
            
    else:
        print("No heatmaps returned")

if __name__ == '__main__':
    debug_relative()
