
import pandas as pd

df = pd.read_csv("data/20252026_raw_game_feeds.csv", nrows=1)
print("Columns:", df.columns.tolist())
