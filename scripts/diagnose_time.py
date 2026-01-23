import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from puck import timing

# We need to look at intervals for a sample of games to be accurate about TIME, not just events.
# Event counts can be misleading (e.g. more shots on PP).

def diagnose():
    season = "20252026"
    # Load a list of games
    try:
        df = pd.read_csv("data/20252026.csv", usecols=['game_id'])
        game_ids = df['game_id'].unique()[:50] # Sample 50 games for speed
    except:
        print("Could not load data/20252026.csv")
        return

    print(f"Diagnosing time distribution for {len(game_ids)} games...")
    
    total_seconds = 0
    state_seconds = {}
    empty_net_seconds = 0
    
    # We want to know:
    # 1. Total Duration
    # 2. Duration of 5v5, 5v4, 4v5 (Non-Empty Net)
    # 3. Duration of Empty Net
    # 4. Duration of Other States (4v4, 3v3, 5v3, etc)
    
    for gid in game_ids:
        # Get all intervals for the game
        # timing.get_game_intervals_cached returns intervals for specific conditions.
        # But we want raw breakdown. 
        # puck.timing might not expose a "get all intervals with labels" easily without iterating conditions.
        
        # Let's iterate widely expected states
        all_states = ['5v5', '5v4', '4v5', '4v4', '3v3', '5v3', '3v5', '4v3', '3v4']
        
        g_total = 0
        
        # Check Empty Net (Global)
        # Note: Empty net can overlap with any state.
        # My script filtered: (game_state == '5v5') & (is_net_empty == 0)
        
        # Strategy: Get total time for each state, then check how much overlap with Empty Net.
        # Actually, timing module allows dict filters.
        
        for state in all_states:
            # 1. Non-Empty Net in this State
            ints_clean = timing.get_game_intervals_cached(gid, season, {'game_state': [state], 'is_net_empty': [0]})
            sec_clean = sum(e-s for s,e in ints_clean)
            
            # 2. Empty Net in this State
            ints_en = timing.get_game_intervals_cached(gid, season, {'game_state': [state], 'is_net_empty': [1]})
            sec_en = sum(e-s for s,e in ints_en)
            
            if state not in state_seconds:
                state_seconds[state] = {'clean': 0.0, 'en': 0.0}
            
            state_seconds[state]['clean'] += sec_clean
            state_seconds[state]['en'] += sec_en
            empty_net_seconds += sec_en
            g_total += (sec_clean + sec_en)
            
        total_seconds += g_total

    print("\n--- Time Distribution (Sample 50 Games) ---")
    avg_total = total_seconds / len(game_ids) / 60
    print(f"Average Total Tracked Time per Game: {avg_total:.2f} min (expect ~60-63)")
    
    print("\nBreakdown by State (Avg Min/Game):")
    other_sum = 0
    
    for state in state_seconds:
        s = state_seconds[state]
        clean_min = (s['clean'] / len(game_ids)) / 60
        en_min = (s['en'] / len(game_ids)) / 60
        total_state = clean_min + en_min
        
        marker = ""
        if state in ['5v5', '5v4', '4v5']:
            marker = "*"
        else:
            other_sum += clean_min # Add clean time of other states to "Other"
        
        print(f"  {state}: {clean_min:.2f} (Clean) + {en_min:.2f} (EN) = {total_state:.2f} Total {marker}")
        
    print(f"\nSummary:")
    processed = (state_seconds['5v5']['clean'] + state_seconds['5v4']['clean'] + state_seconds['4v5']['clean']) / len(game_ids) / 60
    print(f"  Processed (5v5/5v4/4v5 Clean): {processed:.2f} min")
    
    en_total = empty_net_seconds / len(game_ids) / 60
    print(f"  Empty Net (All States):        {en_total:.2f} min")
    
    print(f"  Other States (4v4, 3v3...):    {other_sum:.2f} min")
    
    missing = 60.0 - (processed + en_total + other_sum)
    print(f"  Unaccounted / Gap:             {missing:.2f} min")

if __name__ == "__main__":
    diagnose()
