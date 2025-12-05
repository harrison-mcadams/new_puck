
import time
import pandas as pd
import timing_new
import nhl_api
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_speed():
    # Create a dummy dataframe with some game IDs
    # We'll use a few recent games if possible, or just some hardcoded IDs
    # 2023020001 is a valid game ID (2023-2024 season)
    # Let's use a list of 5 games
    game_ids = [2023020001, 2023020002, 2023020003, 2023020004, 2023020005]
    
    df = pd.DataFrame({'game_id': game_ids})
    
    print(f"Testing demo_for_export with {len(game_ids)} games...")
    start_time = time.time()
    
    # Run demo_for_export
    res = timing_new.demo_for_export(df, condition={'game_state': '5v5'})
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Finished in {duration:.2f} seconds")
    print(f"Average time per game: {duration/len(game_ids):.2f} seconds")
    
    # Check results
    print(f"Result keys: {res.keys()}")
    print(f"Per game count: {len(res.get('per_game', {}))}")

if __name__ == "__main__":
    test_speed()
