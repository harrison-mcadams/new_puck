
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

    # 2. Run xgs_map with 5v4 condition
    print("\n--- Running 5v4 Analysis for PHI ---")
    cond_5v4 = {'game_state': ['5v4'], 'team': 'PHI'}
    
    # xgs_map returns (out_path, heatmaps, filtered_df, summary_stats)
    res_5v4 = analyze.xgs_map(game_id=gid, condition=cond_5v4, return_filtered_df=True, show=False)
    
    if isinstance(res_5v4, tuple) and len(res_5v4) >= 4:
        _, _, df_5v4, summary_5v4 = res_5v4
    else:
        df_5v4 = None
        summary_5v4 = {}
    
    if df_5v4 is not None and not df_5v4.empty:
        print(f"5v4 Events: {len(df_5v4)}")
        
        # Check event owners
        phi_id = None
        if 'home_abb' in df_5v4.columns and df_5v4['home_abb'].iloc[0] == 'PHI':
            phi_id = df_5v4['home_id'].iloc[0]
            print("PHI is Home")
        elif 'away_abb' in df_5v4.columns and df_5v4['away_abb'].iloc[0] == 'PHI':
            phi_id = df_5v4['away_id'].iloc[0]
            print("PHI is Away")
            
        if phi_id:
            phi_events = df_5v4[df_5v4['team_id'] == phi_id]
            opp_events = df_5v4[df_5v4['team_id'] != phi_id]
            print(f"PHI Events: {len(phi_events)}")
            print(f"Opponent Events: {len(opp_events)}")
            
            if not phi_events.empty:
                print("Sample PHI Game States (Raw):")
                print(phi_events['game_state'].unique())
                
                # Simulate add_game_state_relative_column
                df_sim = phi_events.copy()
                df_rel = timing_new.add_game_state_relative_column(df_sim, 'PHI')
                print("Sample PHI Game States (Relative):")
                print(df_rel['game_state_relative_to_team'].unique())
                
            if not opp_events.empty:
                print("Sample Opponent Game States (Raw):")
                print(opp_events['game_state'].unique())
                
                df_sim_opp = opp_events.copy()
                df_rel_opp = timing_new.add_game_state_relative_column(df_sim_opp, 'PHI')
                print("Sample Opponent Game States (Relative):")
                print(df_rel_opp['game_state_relative_to_team'].unique())

        else:
            print("Could not determine PHI ID")
            
    else:
        print("5v4 DF is None or Empty")

if __name__ == '__main__':
    run_repro()
