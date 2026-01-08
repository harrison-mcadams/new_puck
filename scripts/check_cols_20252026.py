
import pandas as pd

df = pd.read_csv("data/20252026.csv", nrows=1)
cols = df.columns.tolist()
print("Columns:", cols)
if 'game_id' in cols and 'season' in cols:
    print("Has game_id and season")
else:
    print("Missing columns")
