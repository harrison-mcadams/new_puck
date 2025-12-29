
import pandas as pd
import requests
import io
import zipfile
import sys
import os

def main():
    game_id = 2025020107
    season = "20242025" # MoneyPuck file year is the start of the season
    mp_year = "2025" # The file is often named by the end year or start year. 20242025 is usually 2024.
    # Actually compare_shots_moneypuck uses 2025 for 20252026. 
    # Let's try 2024 for 2024-2025.
    
    mp_url = f"http://peter-tanner.com/moneypuck/downloads/shots_2025.zip"
    print(f"Downloading MoneyPuck shots from {mp_url}...")
    
    try:
        r = requests.get(mp_url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        df_mp = pd.read_csv(z.open(csv_name))
    except Exception as e:
        print(f"Failed: {e}")
        return

    print("MP Columns:", df_mp.columns.tolist())
    print("MP Head (game_id):", df_mp['game_id'].head().tolist())

    # MP game_id is usually just the suffix of the NHL ID
    # Try both
    game_shots = df_mp[df_mp['game_id'] == game_id]
    if game_shots.empty:
        alt_id = game_id % 1000000
        print(f"No shots for {game_id}, trying {alt_id}...")
        game_shots = df_mp[df_mp['game_id'] == alt_id]
    
    if game_shots.empty:
        print(f"Still no shots found for game {game_id} or {alt_id}")
        return

    # Look for Crouse in P3 around 0:40 (40 seconds into period, or total time)
    # MP 'time' is seconds from start of game. P3 start is 2400s.
    # 0:40 into P3 is 2440s.
    
    crouse_shots = game_shots[game_shots['shooterName'].str.contains('Crouse', na=False)]
    
    print("\nMoneyPuck Data for Lawson Crouse in Game 20107:")
    coord_cols = ['period', 'time', 'shooterName', 'shotDistance', 'xCord', 'yCord', 'xCordAdjusted', 'yCordAdjusted']
    print(crouse_shots[coord_cols].to_string(index=False))

    crouse_shots.to_csv('crouse_mp_data.csv', index=False)
    print("\nFull record saved to crouse_mp_data.csv")

if __name__ == "__main__":
    main()
