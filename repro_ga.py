
import analyze
import pandas as pd
import json

def test_ga_calculation():
    season = '20252026'
    team = 'PHI'
    condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': team}
    
    print(f"Running xgs_map for {team} {season} with condition {condition}...")
    
    # We don't need to save the plot, just get the stats
    out_path, ret_heat, ret_df, summary_stats = analyze.xgs_map(
        season=season,
        condition=condition,
        out_path='test_ga_map.png',
        show=False,
        return_heatmaps=True,
        return_filtered_df=True
    )
    
    print("Summary Stats Keys:", summary_stats.keys())
    print(f"Team Goals: {summary_stats.get('team_goals')}")
    print(f"Other Goals: {summary_stats.get('other_goals')}")
    print(f"Opp Goals (if exists): {summary_stats.get('opp_goals')}")
    
    if summary_stats.get('other_goals', 0) > 0:
        print("SUCCESS: other_goals is populated.")
    else:
        print("FAILURE: other_goals is 0.")

if __name__ == "__main__":
    test_ga_calculation()
