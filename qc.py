
import pandas as pd
import timing
import analyze
import numpy as np

def check_pit_pp_time():
    season = '20252026'
    print(f"Loading season data for {season}...")
    df = timing.load_season_df(season)
    
    if df.empty:
        print("No season data found.")
        return

    # Filter for PIT games
    pit_games = df[(df['home_abb'] == 'PIT') | (df['away_abb'] == 'PIT')]['game_id'].unique()
    print(f"Found {len(pit_games)} games for PIT.")
    
    # Condition for Power Play: 5v4, no empty net
    condition = {'team': 'PIT', 'game_state': ['5v4'], 'is_net_empty': [0]}
    
    print(f"Computing timing for condition: {condition}")
    timing_res = timing.compute_game_timing(df, condition)
    
    total_seconds = timing_res.get('aggregate', {}).get('intersection_seconds_total', 0.0)
    total_minutes = total_seconds / 60.0
    
    print(f"Total PIT PP Time: {total_minutes:.2f} minutes")
    
    # Drill down into individual games to find outliers
    per_game = timing_res.get('per_game', {})
    
    game_times = []
    for gid, res in per_game.items():
        sec = res.get('intersection_seconds', 0.0)
        game_times.append({'game_id': gid, 'minutes': sec / 60.0})
        
    df_times = pd.DataFrame(game_times)
    df_times = df_times.sort_values('minutes', ascending=False)
    
    print("\nTop 10 Games by PP Time:")
    print(df_times.head(10))
    
    # Check specific game 2025020296
    g296 = df_times[df_times['game_id'] == 2025020296]
    if not g296.empty:
        print(f"\nGame 2025020296 PP Time: {g296.iloc[0]['minutes']:.2f} mins")
    else:
        print("\nGame 2025020296 not found in results.")
    
    # Check if we can find a specific game with very high PP time
    if not df_times.empty:
        top_game = df_times.iloc[0]
        if top_game['minutes'] > 20: # Arbitrary high threshold
            print(f"\nInvestigating Game {top_game['game_id']} with {top_game['minutes']:.2f} mins PP time...")
            # We can print more details about this game
            # Maybe the intervals?
            
            # Re-run compute_intervals_for_game with verbose=True for this game
            print(f"--- Debug Output for Game {int(top_game['game_id'])} ---")
            timing.compute_intervals_for_game(int(top_game['game_id']), condition, verbose=True)

if __name__ == "__main__":
    check_pit_pp_time()
