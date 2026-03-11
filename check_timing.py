
import sys
import os
from puck import timing

def check_timing():
    gid = 2025020001
    season = "20252026"
    
    for state in ['5v5', '5v4', '4v5']:
        cond = {'game_state': [state], 'is_net_empty': [0]}
        intervals = timing.get_game_intervals_cached(gid, season, cond)
        total = sum(e - s for s, e in intervals)
        print(f"Game {gid} State {state}: {total:.1f}s ({total/60:.1f} mins)")

if __name__ == "__main__":
    check_timing()
