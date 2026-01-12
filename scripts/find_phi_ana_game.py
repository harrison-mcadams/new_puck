
import json
import glob
from pathlib import Path

DATA_DIR = Path('data/20252026')

def find_game():
    # PHI = 4, ANA = 24
    print("Scanning for PHI vs ANA game...")
    files = list(DATA_DIR.glob('game_*.json'))
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                # Read just the start to get team info usually at top
                # But JSON needs full parse usually. These are small enough (150KB)
                data = json.load(fh)
                
            # Check teams
            home_id = None
            away_id = None
            
            # Helper to get ID
            def get_id(obj):
                if isinstance(obj, dict):
                    return obj.get('id')
                return None
            
            if 'gameData' in data:
                home_id = get_id(data['gameData']['teams']['home'])
                away_id = get_id(data['gameData']['teams']['away'])
            elif 'homeTeam' in data:
                home_id = get_id(data['homeTeam'])
                away_id = get_id(data['awayTeam'])
                
            if {home_id, away_id} == {4, 24}:
                print(f"FOUND MATCH: {f.name}")
                print(f"  Home: {home_id}")
                print(f"  Away: {away_id}")
                return
                
        except Exception:
            continue
            
    print("No matching game found in scanned files.")

if __name__ == "__main__":
    find_game()
