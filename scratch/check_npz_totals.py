import numpy as np
import json
import os
import glob

season = '20252026'
cond = '5v5'
# FLA is team 8 or 13? Let's check team_summary.json for the mapping or just look for the key in NPZ.
# Based on previous output, FLA was team 8? No, FLA was 13 in the partials I checked.
# Wait, let's look at the partials directory again.
# d['team_13_stats'] had FLA Seconds: 2962.0 for game 2025020001.
# 2962 / 60 = 49.3 mins. That's correct for one game!

# So if 2962 is one game, then 82 * 2962 = 242,884 seconds.
# But the user's total was 151,062?
# 151,062 / 2962 = 51 games.
# So maybe they are only aggregating 51 games?
# But they said n_games: 82.

def check_team_totals(target_tid):
    partial_dir = f'data/cache/{season}/partials'
    files = glob.glob(os.path.join(partial_dir, f'*{cond}.npz'))
    
    total_seconds = 0
    game_count = 0
    
    for f in sorted(files):
        try:
            # Extract game ID from filename (e.g. 2025020001_5v5.npz)
            fname = os.path.basename(f)
            gid = fname.split('_')[0]
            data = np.load(f, allow_pickle=True)
            stats_key = f'team_{target_tid}_stats'
            if stats_key in data:
                stats = json.loads(str(data[stats_key]))
                sec = stats.get('team_seconds', 0)
                total_seconds += sec
                game_count += 1
                print(f"Game {gid}: {sec/60:.2f} mins")
        except:
            pass
            
    print(f"Team {target_tid} Total: {total_seconds/60:.2f} mins across {game_count} games.")
    if game_count > 0:
        print(f"Average: {total_seconds/60/game_count:.2f} mins/game")

check_team_totals(13) # FLA
check_team_totals(8)  # Maybe another team?
