import pandas as pd
import os

meta_path = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\metadata.csv"

if os.path.exists(meta_path):
    df = pd.read_csv(meta_path)
    print(f"Total Rows: {len(df)}")
    print("Seasons found:")
    print(df['season'].value_counts())
    print("\nSample Data:")
    print(df.head())
else:
    print("Metadata file not found.")
