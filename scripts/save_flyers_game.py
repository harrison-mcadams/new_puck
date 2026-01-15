import os
import sys
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api
from puck import parse
from puck import data_pipeline

def main():
    logging.basicConfig(level=logging.INFO)
    print("Finding most recent Flyers game...")
    
    # 1. Get Game ID
    try:
        game_id = nhl_api.get_game_id(team='PHI')
        print(f"Most recent Flyers game ID: {game_id}")
    except Exception as e:
        print(f"Error finding game: {e}")
        return

    # 2. Fetch and Parse
    print(f"Fetching and parsing game {game_id}...")
    try:
        # Check cache/fetch
        # _scrape handles fetching and parsing
        # But we can assume we might need to fetch manually if not in season structure
        # Let's use parse._game directly on the feed
        
        feed = nhl_api.get_game_feed(game_id)
        # shifts = nhl_api.get_shifts(game_id) # parse._game handles simple parsing without shifts if needed, or we can fetch
        
        # parse._game expects (game_data, shifts_data) or just game_data depending on signature
        # Let's look at parse.py signature...
        # def _game(game_pd: Dict[str, Any], shifts_pd: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        
        # We'll fetch shifts just in case
        shifts = nhl_api.get_shifts(game_id)
        
        # parse._game expects just the feed dictionary in this version
        events = parse._game(feed)
        df = pd.DataFrame(events)
        print(f"Parsed {len(df)} events.")
        
    except Exception as e:
        print(f"Error parsing game: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Preprocess
    print("Running feature preprocessing...")
    try:
        # Use defaults: training=False, imputation=True, adjust=True
        df_clean = data_pipeline.preprocess_features(
            df, 
            is_training=False,
            apply_imputation=True, 
            apply_arena_adjustments=True,
            verbose=True
        )
        print(f"Processed DataFrame shape: {df_clean.shape}")
        
    except Exception as e:
        print(f"Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Save
    output_dir = 'analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'flyers_debug_{game_id}.csv')
    
    print(f"Saving to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
