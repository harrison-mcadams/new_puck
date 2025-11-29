
import timing
import pandas as pd
import os

def test_load_season_df():
    print("Testing load_season_df...")
    # Create a dummy csv
    os.makedirs('data/20252026', exist_ok=True)
    df = pd.DataFrame({'game_id': [1, 2], 'home_abb': ['PHI', 'PIT']})
    df.to_csv('data/20252026/20252026_df.csv', index=False)
    
    loaded_df = timing.load_season_df('20252026')
    if not loaded_df.empty and len(loaded_df) == 2:
        print("load_season_df passed")
    else:
        print("load_season_df failed")

def test_select_team_game():
    print("Testing select_team_game...")
    df = pd.DataFrame({
        'game_id': [101, 102, 103],
        'home_abb': ['PHI', 'NYR', 'BOS'],
        'away_abb': ['PIT', 'PHI', 'TOR'],
        'home_id': [1, 2, 3],
        'away_id': [4, 1, 5]
    })
    
    # Test by abbreviation
    gids = timing.select_team_game(df, 'PHI')
    # Should be [101, 102]
    if gids == [101, 102]:
        print("select_team_game (abb) passed")
    else:
        print(f"select_team_game (abb) failed: {gids}")

    # Test by ID
    gids_id = timing.select_team_game(df, 1)
    if gids_id == [101, 102]:
        print("select_team_game (id) passed")
    else:
        print(f"select_team_game (id) failed: {gids_id}")

if __name__ == "__main__":
    test_load_season_df()
    test_select_team_game()
