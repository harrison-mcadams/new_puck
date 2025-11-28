
import pandas as pd
import analyze
import sys

# Mock parse.build_mask if needed, or rely on actual implementation
# We will use the actual analyze.py functions

def test_season_filtering():
    season = '20252026'
    team = 'PHI'
    condition = {'game_state': '5v5', 'team': team}
    
    print(f"Running xgs_map for {season} {team} with condition {condition}")
    
    # We need to capture the return value of xgs_map or inspect its internal state.
    # xgs_map returns nothing (it plots), but we can modify it to return stats or we can inspect the printed output?
    # Better: let's import the helper functions from analyze and run the logic step-by-step.
    
    # 1. Load data
    csv_path = f"static/{season}.csv"
    try:
        df_all = pd.read_csv(csv_path)
        print(f"Loaded {len(df_all)} rows from {csv_path}")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Apply condition
    # analyze._apply_condition is defined inside xgs_map? No, it's a helper or defined at module level?
    # Looking at previous view, _apply_condition was indented inside xgs_map? 
    # Let's check analyze.py structure again.
    # If it's inside xgs_map, we can't call it directly.
    # We might need to modify analyze.py to debug or copy the logic.
    
    # Let's assume we can use analyze._apply_condition if it's module level.
    # If not, we will replicate the logic.
    
    # Replicating logic from analyze.py roughly:
    import parse
    
    # Filter for team first (as xgs_map does?)
    # xgs_map calls _apply_condition(df_all)
    
    # We need to see if _apply_condition is available.
    if hasattr(analyze, '_apply_condition'):
        print("Using analyze._apply_condition")
        df_filtered, team_val = analyze._apply_condition(df_all, condition) # Wait, signature might be different
    else:
        print("Replicating filter logic")
        # Replicate basic filtering
        # 1. Team filter
        # 2. Condition filter (5v5)
        
        # Team filter
        tstr = team
        mask_team = (df_all['home_abb'] == tstr) | (df_all['away_abb'] == tstr)
        df_team = df_all[mask_team].copy()
        print(f"Rows after team filter: {len(df_team)}")
        
        # 5v5 filter
        mask_5v5 = df_team['game_state'] == '5v5'
        df_filtered = df_team[mask_5v5].copy()
        print(f"Rows after 5v5 filter: {len(df_filtered)}")
        
    # 3. Calculate xG
    # Check if 'xgs' column exists or needs prediction
    if 'xgs' not in df_filtered.columns:
        print("xgs column missing, mocking it with 0.1 per shot")
        df_filtered['xgs'] = 0.1 # Mock
    
    total_xg = df_filtered['xgs'].sum()
    print(f"Total xG in filtered df: {total_xg}")
    
    # Now let's see what analyze.py actually does by running it
    # We can't easily capture variables from a script run.
    # But we can run analyze.py and check the output logs if we add print statements.
    
if __name__ == "__main__":
    test_season_filtering()
