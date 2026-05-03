import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from puck import timing, nhl_api
import pandas as pd

# Set up logging to see the fallback triggers
logging.basicConfig(level=logging.INFO)

def verify_game(game_id):
    print(f"\n--- Verifying Game {game_id} ---")
    
    # We want to force a re-fetch to see the logic in action
    # But timing._get_shifts_df uses local caching.
    # We'll bypass the disk cache by temporarily renaming it or just calling the logic directly.
    
    # Actually, we can just call get_shifts_from_nhl_html directly to verify the dual-fetch
    print("Testing get_shifts_from_nhl_html (Direct)...")
    html_res = nhl_api.get_shifts_from_nhl_html(game_id, debug=True)
    if html_res and html_res.get('all_shifts'):
        df = pd.DataFrame(html_res['all_shifts'])
        teams = df['team_id'].unique()
        print(f"HTML Fallback found {len(df)} shifts for teams: {teams}")
        if len(teams) >= 2:
            print("SUCCESS: Both teams found in HTML.")
        else:
            print("FAILURE: Only one team found in HTML.")
    else:
        print("FAILURE: HTML fallback returned no data.")

    print("\nTesting timing._get_shifts_df (Orchestration)...")
    # This will check API first, then trigger fallback if team count < 2
    df_final = timing._get_shifts_df(game_id)
    if df_final is not None:
        teams = df_final['team_id'].unique()
        print(f"Final DataFrame has {len(df_final)} shifts for teams: {teams}")
        if len(teams) >= 2:
            print("SUCCESS: Orchestration recovered both teams.")
        else:
            print("FAILURE: Orchestration failed to recover both teams.")
    else:
        print("FAILURE: timing._get_shifts_df returned None.")

if __name__ == "__main__":
    # Test the specific game identified earlier (2025020352)
    verify_game(2025020352)
    
    # Test a historical game for regression
    verify_game(2018020001)
