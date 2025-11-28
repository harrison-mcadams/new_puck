import analyze
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

print("Running debug analysis for ANA...")
try:
    # Run season_analysis to get the raw data (we'll intercept it or just run it and check the saved files if we can, 
    # but better to call xgs_map directly or modify season_analysis to print debug info.
    # Actually, let's just call xgs_map directly to see what it returns.
    
    print("Calling xgs_map directly...")
    condition = {'team': 'ANA', 'game_state': ['5v5'], 'is_net_empty': [0]}
    out_path, ret_heat, ret_df, summary_stats = analyze.xgs_map(
        season='20252026',
        condition=condition,
        return_heatmaps=True
    )
    
    team_map = ret_heat.get('team')
    other_map = ret_heat.get('other')
    
    print(f"Team Map Shape: {team_map.shape}")
    print(f"Other Map Shape: {other_map.shape}")
    
    # Check where the data is located (Left vs Right)
    # Grid is 200 x 85 (approx)
    # Left is x < 100 (in array indices) or x < 0 in coordinates?
    # analyze.py uses:
    # gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
    # gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
    # Array shape is (86, 201) usually (ny, nx)
    
    # Let's sum up the left and right halves
    mid_x = team_map.shape[1] // 2
    
    team_left_sum = np.nansum(team_map[:, :mid_x])
    team_right_sum = np.nansum(team_map[:, mid_x:])
    
    other_left_sum = np.nansum(other_map[:, :mid_x])
    other_right_sum = np.nansum(other_map[:, mid_x:])
    
    print(f"Team Map Left Sum: {team_left_sum:.4f}")
    print(f"Team Map Right Sum: {team_right_sum:.4f}")
    print(f"Other Map Left Sum: {other_left_sum:.4f}")
    print(f"Other Map Right Sum: {other_right_sum:.4f}")
    
    if team_left_sum > team_right_sum:
        print("Team Map is Left-Oriented")
    else:
        print("Team Map is Right-Oriented")
        
    if other_left_sum > other_right_sum:
        print("Other Map is Left-Oriented")
    else:
        print("Other Map is Right-Oriented")

    # Load League Baseline
    print("\nLoading League Baseline...")
    league_res = analyze.league(season='20252026', mode='load')
    league_baseline = league_res['league_baseline_map']
    
    league_left_sum = np.nansum(league_baseline[:, :mid_x])
    league_right_sum = np.nansum(league_baseline[:, mid_x:])
    
    print(f"League Baseline Left Sum: {league_left_sum:.4f}")
    print(f"League Baseline Right Sum: {league_right_sum:.4f}")
    
    if league_left_sum > league_right_sum:
        print("League Baseline is Left-Oriented")
    else:
        print("League Baseline is Right-Oriented")

except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
