import pandas as pd

raw_feed_csv = r'data/20252026_raw_game_feeds.csv'
target_game = 2025020008

print(f"Checking for Game {target_game} in {raw_feed_csv}...")

try:
    # Read just game_id column first if possible, or iterate chunks
    found = False
    for chunk in pd.read_csv(raw_feed_csv, chunksize=1000):
        if target_game in chunk['game_id'].values:
            print(f"Found Game {target_game}!")
            # Get the row
            row = chunk[chunk['game_id'] == target_game].iloc[0]
            print(f"Feed string length: {len(row['feed'])}")
            print(f"First 100 chars of feed: {row['feed'][:100]}")
            found = True
            break
            
    if not found:
        print(f"Game {target_game} NOT found in CSV.")

except Exception as e:
    print(f"Error: {e}")
