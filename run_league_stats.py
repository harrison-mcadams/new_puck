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
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    # REGENERATE_PLOTS_ONLY:
    #   False (Default): Full Re-computation.
    #       - Loads raw season dataframe (e.g. data/20252026/20252026.csv).
    #       - Re-calculates game timing, xG values, and spatial maps for every team.
    #       - Saves new intermediate files (.npz) and summary stats.
    #       - Generates all plots.
    #       - USE WHEN: You have new game data or have changed calculation logic (e.g. xG model, timing).
    #
    #   True: Plotting Only (Fast).
    #       - Skips all heavy calculation.
    #       - Loads existing intermediate data from static/league/{season}/{condition}/intermediates/.
    #       - Re-runs only the plotting functions (heatmaps, scatter plots).
    #       - USE WHEN: You are tweaking plot aesthetics (colors, labels, text, titles) and the data is already correct.
    REGENERATE_PLOTS_ONLY = False 
    
    # target_teams:
    #   None (Default): Process all teams found in the season data.
    #   ['ANA', 'BOS']: Process only specific teams. Useful for quick testing.
    #   NOTE: For relative maps, you generally need a full league run to get a valid league baseline.
    #         If running a subset, the baseline will be calculated from ONLY that subset.
    target_teams = None
    
    # UPDATE_DATA:
    #   False (Default): Use existing data on disk.
    #   True: Force a fresh fetch of season data from the NHL API before running analysis.
    #         - Calls parse._scrape with use_cache=False.
    #         - Updates data/{season}/{season}.csv.
    #         - Useful for getting the latest games.
    UPDATE_DATA = False

    def update_data(season: str):
        """
        Helper to force a fresh fetch of season data from the NHL API.
        Updates the season CSV and other necessary files.
        """
        print(f"update_data: Fetching fresh data for season {season}...")
        import parse
        
        # parse._scrape handles fetching raw feeds, processing them, and saving the elaborated CSV.
        # We set use_cache=False to force fresh API calls.
        # We set process_elaborated=True and save_elaborated=True to generate the season.csv.
        parse._scrape(
            season=season,
            out_dir='data',
            use_cache=False,
            process_elaborated=True,
            save_elaborated=True,
            return_elaborated_df=False,
            verbose=True
        )
        print(f"update_data: Completed update for {season}.")

    if UPDATE_DATA:
        update_data(season)
    
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
            
            # 3. Scatter Plot
            print(f"Generating scatter plot for {cond_name}...")
            # We need the summary list again. It might have been updated by season() if we were tracking it,
            # but season() writes to disk.
            # analyze.generate_scatter_plot reads from disk if we pass summary_list?
            # No, generate_scatter_plot takes summary_list as arg.
            # league_res['summary'] has the initial summary.
            # But season() updates stats (like percentiles).
            # However, scatter plot only needs xGF/60 and xGA/60 which are in league_res['summary'] usually?
            # Wait, league_res['summary'] comes from analyze.league().
            # analyze.league() computes them.
            # So we can use league_res['summary'].
            analyze.generate_scatter_plot(league_res.get('summary', []), f'static/league/{season}/{cond_name}', cond_name)
                
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
