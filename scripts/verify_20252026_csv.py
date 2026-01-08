
import pandas as pd
import os

path = "data/20252026.csv"
if os.path.exists(path):
    df = pd.read_csv(path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("Sample row:", df.iloc[0].to_dict())
else:
    print("File not found")
