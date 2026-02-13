
import pandas as pd

def main():
    df = pd.read_csv("data/20252026.csv")
    
    # Filter for EDM games
    edm_games = df[(df['home_abb'] == 'EDM') | (df['away_abb'] == 'EDM')]['game_id'].unique()
    edm_df = df[df['game_id'].isin(edm_games)]
    
    # Filter for Goals
    goals = edm_df[edm_df['event'] == 'goal'].copy()
    
    # Determine if goal is FOR EDM or AGAINST EDM
    # team_name column might indicate the scoring team
    print("Columns:", goals.columns.tolist())
    
    # Columns have: team_id, home_id, away_id, home_abb, away_abb
    # For goals, team_id should be the scoring team
    # We need to find EDM's team_id first
    
    # Get EDM team_id from a game where EDM is home
    edm_home_game = edm_df[edm_df['home_abb'] == 'EDM'].iloc[0] if len(edm_df[edm_df['home_abb'] == 'EDM']) > 0 else None
    if edm_home_game is not None:
        edm_team_id = edm_home_game['home_id']
        print(f"EDM Team ID: {edm_team_id}")
    else:
        edm_away_game = edm_df[edm_df['away_abb'] == 'EDM'].iloc[0]
        edm_team_id = edm_away_game['away_id']
        print(f"EDM Team ID (from away): {edm_team_id}")
    
    # Filter for EDM goals
    edm_goals = goals[goals['team_id'] == edm_team_id].copy()
    
    print(f"\n--- EDM Goals in {len(edm_games)} Games ---")
    print(f"Total EDM Goals (All Situations): {len(edm_goals)}")
    
    # Group by game_state
    print("\nEDM Goals by game_state Label:")
    print(edm_goals['game_state'].value_counts())
    
    # Group by game_state
    print("\nEDM Goals by game_state Label:")
    print(edm_goals['game_state'].value_counts())
    
    # Now check: How many goals are in the PP time windows but labeled 5v5?
    # Need to use puck.timing to get the 5v4 intervals
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from puck import timing
    
    # Compute time column
    def time_to_sec(x):
        try:
            m, s = x.split(':')
            return int(m)*60 + int(s)
        except: return 0
    edm_df['period_seconds'] = edm_df['period_time'].apply(time_to_sec)
    edm_df['total_seconds'] = (edm_df['period'] - 1) * 1200 + edm_df['period_seconds']
    
    # Check 5v5 labeled goals that occur during 5v4 intervals
    mislabeled_goals = 0
    mislabeled_details = []
    
    for gid in edm_games:
        g_df = edm_df[edm_df['game_id'] == gid]
        if g_df.empty: continue
        
        home = g_df['home_abb'].iloc[0]
        # If EDM is Home, 5v4 state is '5v4'. If Away, '4v5'.
        target_state = '5v4' if home == 'EDM' else '4v5'
        
        try:
            cond = {'game_state': [target_state], 'is_net_empty': [0]}
            intervals = timing.get_game_intervals_cached(gid, "20252026", cond)
        except:
            continue
            
        g_goals = g_df[(g_df['event'] == 'goal') & (g_df['team_id'] == edm_team_id)]
        
        for _, row in g_goals.iterrows():
            t = row['total_seconds']
            labeled_state = row['game_state']
            
            # Check if this goal time falls within a 5v4 interval
            in_pp = any(s <= t < e for s, e in intervals)
            
            if in_pp and labeled_state not in ['5v4', '4v5']:
                mislabeled_goals += 1
                mislabeled_details.append({
                    'game_id': gid,
                    'time': row['period_time'],
                    'labeled': labeled_state,
                    'should_be': target_state
                })
    
    print(f"\nMislabeled PP Goals (in valid 5v4 interval but labeled 5v5): {mislabeled_goals}")
    for d in mislabeled_details[:10]:
        print(f"  Game {d['game_id']} at {d['time']}: Labeled '{d['labeled']}', Should be '{d['should_be']}'")

if __name__ == "__main__":
    main()
