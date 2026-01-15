
import pandas as pd
import os

path = 'data/20252026/20252026_df.csv'
if os.path.exists(path):
    df = pd.read_csv(path, nrows=5)
    print(f"Columns in {path}:")
    print(df.columns.tolist())
    if 'xgs' in df.columns:
        print("ALERT: 'xgs' column found!")
    else:
        print("Clean: No 'xgs' column.")
else:
    print(f"File not found: {path}")
