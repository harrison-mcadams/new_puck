import analyze
import matplotlib
import matplotlib.pyplot as plt
import os

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

if __name__ == '__main__':
    season = '20252026'
    
    # Define conditions for analysis
    conditions = {
        '5v5': {'game_state': ['5v5'], 'is_net_empty': [0]},
        '5v4': {'game_state': ['5v4'], 'is_net_empty': [0]},
        '4v5': {'game_state': ['4v5'], 'is_net_empty': [0]}
    }
    
    print(f"Running season analysis for {season} with conditions: {list(conditions.keys())}")
    
    REGENERATE_PLOTS_ONLY = False 
    target_teams = None
    
    # Get team list
    import json
    teams_to_process = []
    if target_teams:
        teams_to_process = target_teams
    else:
        try:
            with open('static/teams.json', 'r') as f:
                teams_data = json.load(f)
            teams_to_process = [t.get('abbr') for t in teams_data if 'abbr' in t]
        except Exception as e:
            print(f"Warning: failed to load teams.json: {e}")
            # Fallback will happen in analyze.league if we pass None, but we need list for season loop
            # We can let analyze.league find them, then read the summary?
            # Or just rely on analyze.league to return the list of teams it processed?
            pass

    try:
        for cond_name, cond in conditions.items():
            print(f"\n=== Processing Condition: {cond_name} ===")
            
            # 1. League Baseline & Summary
            # This generates baseline.npy and team_summary.json/csv
            # It also saves intermediate team maps to disk
            league_res = analyze.league(
                season=season,
                condition=cond,
                mode='compute' if not REGENERATE_PLOTS_ONLY else 'load',
                teams=target_teams
            )
            
            if not league_res:
                print(f"Failed to get league results for {cond_name}")
                continue
                
            # If teams_to_process was empty, populate from league results
            if not teams_to_process:
                # league_res['summary'] is a list of dicts
                summary = league_res.get('summary', [])
                teams_to_process = [r['team'] for r in summary if r.get('team') != 'League']
                teams_to_process = sorted(list(set(teams_to_process)))
            
            print(f"--- Team Analysis ({cond_name}) for {len(teams_to_process)} teams ---")
            
            # 2. Per-Team Analysis (Plots)
            for i, team in enumerate(teams_to_process):
                if i % 5 == 0:
                    import gc
                    gc.collect()
                    
                # print(f"Processing {team}...")
                analyze.season(
                    season=season,
                    team=team,
                    condition=cond,
                    league_data=league_res
                )
                
        print("\n=== Generating Special Teams Plots ===")
        # We need the base output directory where 5v4/4v5 folders are
        # analyze.season uses static/league/{season}/{cond_name} by default
        # So base is static/league/{season}
        base_out_dir = f'static/league/{season}'
        analyze.generate_special_teams_plot(season, teams_to_process, base_out_dir)
        
        print("Season analysis complete.")
        
    except Exception as e:
        print(f"Error running season analysis: {e}")
        import traceback
        traceback.print_exc()
