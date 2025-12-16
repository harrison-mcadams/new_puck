
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import nhl_api

def test_season_fetch():
    season = "20232024"
    
    print(f"Testing fetch for season {season}")
    
    print("1. Testing DEFAULT (team='PHI')...")
    games_phi = nhl_api.get_season(season=season)
    print(f"   Matches: {len(games_phi)}")
    
    print("2. Testing TEAM='all'...")
    games_all = nhl_api.get_season(season=season, team='all')
    print(f"   Matches: {len(games_all)}")
    
    if len(games_all) > 1000:
        print("\nSUCCESS: 'all' returns league-wide schedule.")
    else:
        print("\nFAILURE: 'all' returned too few games.")

if __name__ == "__main__":
    test_season_fetch()
