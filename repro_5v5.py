
import pandas as pd
import analyze
import timing_new
import nhl_api
import sys

def run_repro():
    # 1. Pick a game (PHI vs someone recent)
    try:
        gid = nhl_api.get_game_id(team='PHI')
        print(f"Using game {gid}")
    except:
        print("Could not find PHI game, using hardcoded if available or failing")
        return

    # 2. Run xgs_map with 5v5 condition
    print("\n--- Running 5v5 Analysis ---")
    cond_5v5 = {'game_state': ['5v5'], 'is_net_empty': [0]}
    # xgs_map returns (out_path, heatmaps, filtered_df, summary_stats)
    res_5v5 = analyze.xgs_map(game_id=gid, condition=cond_5v5, return_filtered_df=True, show=False)
    
    if isinstance(res_5v5, tuple) and len(res_5v5) >= 4:
        _, _, df_5v5, summary_5v5 = res_5v5
    else:
        df_5v5 = None
        summary_5v5 = {}
    
    if df_5v5 is not None:
        print(f"5v5 Events: {len(df_5v5)}")
        print(f"5v5 xG (Team): {summary_5v5.get('team_xgs')}")
        print(f"5v5 xG (Other): {summary_5v5.get('other_xgs')}")
    else:
        print("5v5 DF is None")

    # 3. Run xgs_map with NO condition (All Situations)
    print("\n--- Running All Situations Analysis ---")
    res_all = analyze.xgs_map(game_id=gid, condition=None, return_filtered_df=True, show=False)
    
    if isinstance(res_all, tuple) and len(res_all) >= 4:
        _, _, df_all, summary_all = res_all
    else:
        df_all = None
        summary_all = {}
    
    if df_all is not None:
        print(f"All Events: {len(df_all)}")
        print(f"All xG (Team): {summary_all.get('team_xgs')}")
        print(f"All xG (Other): {summary_all.get('other_xgs')}")

    # 4. Check timing intervals
    print("\n--- Checking Timing Intervals ---")
    intervals = timing_new.compute_intervals_for_game(gid, cond_5v5, verbose=True)
    
    # 5. Check if add_game_state_relative_column exists
    print(f"\nHas add_game_state_relative_column: {hasattr(timing_new, 'add_game_state_relative_column')}")

if __name__ == '__main__':
    run_repro()
