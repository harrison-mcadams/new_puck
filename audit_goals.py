
import pandas as pd
import numpy as np

def audit():
    df = pd.read_csv('data/20252026.csv')
    
    # Identify unique teams
    teams = df.dropna(subset=['team_id'])['team_id'].unique()
    
    print(f"{'Team':<5} | {'State':<5} | {'Seconds':<10} | {'Goals':<5} | {'G/60':<5}")
    print("-" * 40)
    
    for tid in [16, 13, 8]: # CHI, FLA, PHI
        abb = df[df['team_id'] == tid]['team_id'].iloc[0] # Just use ID for now
        
        for state in ['5v5', '5v4', '4v5']:
            # Time calculation is tricky without shifts, so let's just look at counts
            mask = (df['team_id'] == tid) & (df['game_state'] == state)
            goals = df[mask & (df['event'] == 'goal')]
            
            # For seconds, we need to estimate from the whole df
            # But the team_stats_summary.json already has the seconds.
            # Let's just trust the goal counts for now.
            print(f"{tid:<5} | {state:<5} | Goals: {len(goals)}")

    # Specific check for CHI (16) 5v4 Goals
    chi_pp_goals = df[(df['team_id'] == 16) & (df['game_state'] == '5v4') & (df['event'] == 'goal')]
    print(f"\nCHI (16) 5v4 Goals: {len(chi_pp_goals)}")
    
    # Are there 4v5 goals where CHI has 5?
    # If CHI is Away (id 16) and game_state is 4v5, then Home has 4, Away has 5.
    chi_away_pp_goals = df[(df['team_id'] == 16) & (df['game_state'] == '4v5') & (df['event'] == 'goal')]
    print(f"CHI (16) 4v5 Goals: {len(chi_away_pp_goals)}")

if __name__ == "__main__":
    audit()
