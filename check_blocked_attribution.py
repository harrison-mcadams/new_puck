import pandas as pd
import numpy as np
import os

def check_season(season_path):
    if not os.path.exists(season_path):
        print(f"Path {season_path} not found.")
        return
    
    df = pd.read_csv(season_path)
    blocks = df[df['event'].str.lower() == 'blocked-shot'].head(5)
    
    if blocks.empty:
        print(f"No blocked shots found in {season_path}")
        return

    print(f"\n--- Checking {season_path} ---")
    for idx, row in blocks.iterrows():
        # In recent API:
        # 'team_id' is the performer.
        # 'home_id' and 'away_id' are the teams.
        # We need to see if the player_id (shooter) matches the team_id.
        # And if the 'event' description (if we had it) matches.
        
        # Actually, let's just look at 'is_home' vs 'x'.
        # If is_home=1 and x > 70 (approx), they are in offensive zone.
        # If team_id was the blocker, they would be in defensive zone (x < -70).
        
        # Wait, the orientation standardization already happened in these CSVs?
        # Let's check raw 'x' if possible, but usually these are already standard.
        
        print(f"Event: {row['event']}, Player: {row.get('player_name', 'N/A')}, TeamID: {row.get('team_id', 'N/A')}, X: {row['x']:.1f}, Dist: {row.get('distance', -1):.1f}")
        
    # Heuristic: If 100% of blocks have X > 0 after pipeline processing, it means they are oriented as attackers.
    # If the pipeline ORIENTED them as attackers but they were actually blockers, x would be negative or distances would be weird.
    
    print(f"Avg X for blocks: {df[df['event'].str.lower() == 'blocked-shot']['x'].mean():.2f}")

# Check 20202021
check_season('data/20202021/20202021_df.csv')

# Check 20252026 if exists
check_season('data/20252026/20252026_df.csv')
