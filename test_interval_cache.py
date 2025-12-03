
import timing
import os
import json
import shutil

def test_interval_cache():
    game_id = 2025020211
    season = '20252026'
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # Ensure clean state
    cache_dir = f"data/{season}/game_intervals"
    cache_file = f"{cache_dir}/{game_id}.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    print(f"Testing interval cache for game {game_id}...")
    
    # 1. First Call (Should Compute and Cache)
    intervals = timing.get_game_intervals_cached(game_id, season, condition)
    print(f"Computed intervals: {len(intervals)} segments")
    
    if not os.path.exists(cache_file):
        print("FAIL: Cache file not created.")
        return
        
    with open(cache_file, 'r') as f:
        data = json.load(f)
        
    if '5v5' not in data:
        print("FAIL: '5v5' key missing from cache.")
        return
        
    print("PASS: Cache file created with '5v5' key.")
    
    # 2. Second Call (Should Load from Cache)
    # We can verify this by modifying the cache manually and seeing if it loads the modified value
    data['5v5'] = [[0, 100]] # Fake data
    with open(cache_file, 'w') as f:
        json.dump(data, f)
        
    intervals_2 = timing.get_game_intervals_cached(game_id, season, condition)
    if intervals_2 == [[0, 100]]:
        print("PASS: Loaded from cache.")
    else:
        print(f"FAIL: Did not load from cache. Got {intervals_2}")

if __name__ == "__main__":
    test_interval_cache()
