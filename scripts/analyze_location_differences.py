
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def analyze_locations():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Overall Stats
    # dist_diff = distance (Local) - shotDistance (MP)
    # Positive = Local thinks it's further away
    # Negative = Local thinks it's closer
    
    print(f"Total Shots: {len(df)}")
    print(f"Mean Location Diff (Local - MP): {df['dist_diff'].mean():.2f} ft")
    print(f"MAE Location Diff: {df['dist_diff'].abs().mean():.2f} ft")
    print(f"Median Abs Diff: {df['dist_diff'].abs().median():.2f} ft")
    
    # 2. Large Discrepancies
    threshold = 10.0 # ft
    bad_locs = df[df['dist_diff'].abs() > threshold].copy()
    print(f"\nShots with > {threshold}ft Disagreement: {len(bad_locs)} ({len(bad_locs)/len(df)*100:.2f}%)")
    
    # 3. By Arena (Home Team)
    # We merged MP data which has 'team' (shooter team). We need to know the *Home* team to identify Arena.
    # In MP columns we requested 'isHome'. Let's see if we have it. 
    # If not, we can infer from our local data if we had home_id... 
    # The deep dive CSV has 'team_id' (shooter) but maybe not home/away.
    # Actually, we can assume game_id -> Home Team if we load the schedule/season df, or just look at 'team' if 'isHome' is true?
    # Wait, the deep dive CSV columns are: 
    # ['game_id', 'period', 'game_seconds_calc', 'team_id', 'event', 'xgs', 'player_id', 'x', 'y', 'distance', 'angle_deg', 'is_net_empty', 'is_rebound', 'shot_type', 'id_local', 'sec', 'time', 'team', 'xGoal', 'shotID', 'shooterName', 'shotType', 'shotDistance', 'shotAngle', 'xCord', 'yCord', 'emptyNet', 'diff', 'abs_diff', 'dist_diff']
    # We don't have 'isHome' or 'home_team' readily available in this CSV. 
    # BUT, we can use the 'game_id' to look up the home team if we load the parsed season data.
    
    print("\n--- Loading Season Data for Arena Context ---")
    try:
        df_season = pd.read_csv(os.path.join(config.DATA_DIR, '20252026.csv'))
        # Create map of game_id -> home_abb
        # Local GameID is full (202502...)
        # MP GameID is short (20...)
        # The CSV has normalized game_id (short). 
        # We need to normalize season df game_id too.
        df_season['game_id_short'] = df_season['game_id'].astype(int) % 1000000
        game_home_map = df_season.drop_duplicates('game_id_short').set_index('game_id_short')['home_abb'].to_dict()
        
        df['home_team'] = df['game_id'].map(game_home_map)
    except Exception as e:
        print(f"Could not load season context: {e}")
        df['home_team'] = 'Unknown'

    # Group by Arena (Home Team)
    print("\n--- Mean Abs Error by Arena (Home Team) ---")
    arena_stats = df.groupby('home_team')['dist_diff'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    print(arena_stats.head(10))
    
    # 4. By Event Type / Shot Type
    print("\n--- Mean Abs Error by Shot Type ---")
    print(df.groupby('shotType')['dist_diff'].apply(lambda x: x.abs().mean()).sort_values(ascending=False))
    
    print("\n--- Mean Bias (Signed Diff) by Shot Type ---")
    print(df.groupby('shotType')['dist_diff'].mean().sort_values(ascending=False))
    
    # 5. Extreme Examples
    cols = ['game_id', 'period', 'sec', 'home_team', 'shooterName', 'event', 'shotType', 'distance', 'shotDistance', 'dist_diff']
    print(f"\n--- Extreme Examples (Diff > 30ft) ---")
    extreme = df[df['dist_diff'].abs() > 30.0].sort_values('dist_diff', ascending=False)
    if not extreme.empty:
        print(extreme[cols].head(10).to_string(index=False))
        
    # Check if systematic coordinate issue?
    # MP x,y vs Local x,y
    # Local x,y are in 'x', 'y'. MP are 'xCord', 'yCord'.
    # MP coords are often 100x42 ? No, usually standard.
    # Check correlation of x and y
    print("\n--- Coordinate Checks ---")
    # Clean NaNs
    df_coords = df.dropna(subset=['x', 'y', 'xCord', 'yCord'])
    print(f"X Correlation: {df_coords['x'].corr(df_coords['xCord']):.4f}")
    print(f"Y Correlation: {df_coords['y'].corr(df_coords['yCord']):.4f}")
    
    # Check if MP xCord needs flipping for away teams? 
    # Or if Local does? 
    # Usually both are normalized to attacking zone (0-100 or similar).
    
    # Check if there's a flip issue
    # If X corr is negative, then one is flipped.
    
if __name__ == "__main__":
    analyze_locations()
