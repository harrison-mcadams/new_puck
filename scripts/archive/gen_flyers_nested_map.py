
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck.nhl_api import get_game_id
from puck.fit_nested_xgs import NestedXGClassifier

def main():
    # 1. Get most recent Flyers game
    print("Finding most recent Flyers game...")
    try:
        # method='most_recent' defaults to current/past games
        game_id = get_game_id(team='PHI', method='most_recent')
        print(f"Most recent game ID: {game_id}")
    except Exception as e:
        print(f"Failed to find game: {e}")
        return

    # 2. Generate Map with Nested Model
    # Since we updated analyze.py default, we don't strictly need to pass model_path,
    # but we can explicitly pass it or rely on the default we just verified.
    # We'll rely on the default to prove it works as configured.
    print(f"Generating map for Game {game_id} using default (nested) model...")
    
    try:
        # show=False to avoid popping up window, we just want to save
        # out_path defaults to analysis/game_{game_id}_xgs.png if not specified? 
        # Let's specify one to be sure.
        out_path = f"analysis/game_{game_id}_nested_xgs.png"
        
        analyze.xgs_map(
            season='20252026', 
            game_id=game_id, 
            out_path=out_path,
            model_path='analysis/xgs/xg_model_nested.joblib',
            condition={'team': 'PHI'},
            show=False
        )
        print(f"Map saved to {out_path}")
    except Exception as e:
        print(f"Error generating map: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
