import analyze
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

if __name__ == '__main__':
    season = '20252026'
    
    # Define conditions for analysis
    # Each key is a condition name (used for directory structure)
    # Each value is the condition dictionary passed to analyze.season -> league/xgs_map
    conditions = {
        '5v5': {'game_state': ['5v5'], 'is_net_empty': [0]},
        '5v4': {'game_state': ['5v4'], 'is_net_empty': [0]},
        '4v5': {'game_state': ['4v5'], 'is_net_empty': [0]}
    }
    
    print(f"Running season analysis for {season} with conditions: {list(conditions.keys())}")
    
    # analyze.season now handles looping over conditions, fetching baselines,
    # generating team maps, relative maps, summary.json, and scatter plots.
    # It saves results to static/league_stats/{season}/{condition_name}/
    
    try:
        results = analyze.season(
            season=season,
            conditions=conditions,
            out_dir=f'static/league/{season}'
        )
        print("Season analysis complete.")
        
    except Exception as e:
        print(f"Error running season analysis: {e}")
        import traceback
        traceback.print_exc()

