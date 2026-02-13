"""
Deep diagnostic: Compare penalty events to puck.timing intervals
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def main():
    df = pd.read_csv("data/20252026.csv")
    
    # Pick a specific EDM game to analyze in detail
    # Use game 2025020006 which we analyzed earlier
    gid = 2025020006
    
    g_df = df[df['game_id'] == gid].copy()
    home = g_df['home_abb'].iloc[0]
    away = g_df['away_abb'].iloc[0]
    print(f"Game {gid}: {home} vs {away}")
    
    # Compute time
    def time_to_sec(x):
        try:
            m, s = x.split(':')
            return int(m)*60 + int(s)
        except: return 0
    g_df['period_seconds'] = g_df['period_time'].apply(time_to_sec)
    g_df['total_seconds'] = (g_df['period'] - 1) * 1200 + g_df['period_seconds']
    
    # 1. Find all penalty events
    penalties = g_df[g_df['event'] == 'penalty']
    print(f"\n=== Penalty Events ({len(penalties)}) ===")
    if not penalties.empty:
        print(penalties[['period', 'period_time', 'total_seconds', 'game_state']].head(20))
    
    # 2. Get 5v4 intervals from puck.timing
    print("\n=== 5v4 Intervals from puck.timing ===")
    try:
        cond_5v4 = {'game_state': ['5v4'], 'is_net_empty': [0]}
        intervals_5v4 = timing.get_game_intervals_cached(gid, "20252026", cond_5v4)
        print(f"Found {len(intervals_5v4)} intervals for 5v4:")
        for i, (s, e) in enumerate(intervals_5v4[:10]):
            print(f"  {i+1}. {s:.1f}s - {e:.1f}s (duration: {e-s:.1f}s)")
    except Exception as ex:
        print(f"Error getting 5v4 intervals: {ex}")
        intervals_5v4 = []
    
    print("\n=== 4v5 Intervals from puck.timing ===")
    try:
        cond_4v5 = {'game_state': ['4v5'], 'is_net_empty': [0]}
        intervals_4v5 = timing.get_game_intervals_cached(gid, "20252026", cond_4v5)
        print(f"Found {len(intervals_4v5)} intervals for 4v5:")
        for i, (s, e) in enumerate(intervals_4v5[:10]):
            print(f"  {i+1}. {s:.1f}s - {e:.1f}s (duration: {e-s:.1f}s)")
    except Exception as ex:
        print(f"Error getting 4v5 intervals: {ex}")
        intervals_4v5 = []
    
    # 3. Find all goals in this game
    goals = g_df[g_df['event'] == 'goal']
    print(f"\n=== Goals in Game ({len(goals)}) ===")
    print(goals[['period', 'period_time', 'total_seconds', 'game_state', 'team_id']].head(20))
    
    # 4. Check which goals fall within 5v4 or 4v5 intervals
    print("\n=== Goal Classification ===")
    for _, goal in goals.iterrows():
        t = goal['total_seconds']
        label = goal['game_state']
        
        in_5v4 = any(s <= t < e for s, e in intervals_5v4)
        in_4v5 = any(s <= t < e for s, e in intervals_4v5)
        
        status = []
        if in_5v4: status.append("IN_5v4_INTERVAL")
        if in_4v5: status.append("IN_4v5_INTERVAL")
        if not status: status.append("NOT_IN_PP_INTERVAL")
        
        print(f"  {goal['period_time']} (t={t:.0f}s) Label={label} -> {', '.join(status)}")
    
    # 5. Check the raw game_state distribution in the game
    print("\n=== Game State Distribution ===")
    print(g_df['game_state'].value_counts())

if __name__ == "__main__":
    main()
