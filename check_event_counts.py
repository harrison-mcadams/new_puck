import pandas as pd
import glob
import os

def check_events():
    csv_files = glob.glob('data/20242025/20242025_df.csv')
    if not csv_files:
        print("No CSV found.")
        return
    
    df = pd.read_csv(csv_files[0])
    if 'last_event_type' not in df.columns:
        print("Column last_event_type missing.")
        return
        
    counts = df['last_event_type'].value_counts()
    print("TOP 50 LAST EVENT TYPES:")
    for evt, count in counts.head(50).items():
        print(f"{evt}: {count}")

if __name__ == "__main__":
    check_events()
