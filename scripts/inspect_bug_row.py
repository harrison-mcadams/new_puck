
import pandas as pd
import sys

def main():
    try:
        df = pd.read_csv("analysis/flyers_processed_sample.csv")
        print("Loaded CSV.")
        
        # Filter for blocked shots
        mask = df['event'] == 'blocked-shot'
        df_blk = df[mask]
        
        if df_blk.empty:
            print("No blocked shots found.")
            return
            
        print(f"Found {len(df_blk)} blocked shots.")
        
        # Select relevant columns if they exist
        cols = ['period', 'period_time', 'event', 'team_id', 'home_id', 'away_id', 'home_abb', 'away_abb']
        existing_cols = [c for c in cols if c in df.columns]
        
        # Print first few to check types
        print("\n--- Inspecting first 5 blocked shots ---")
        print(df_blk[existing_cols].head())
        
        print("\n--- Data Types ---")
        print(df_blk[existing_cols].dtypes)
        
        # Specific check for 19:25 if possible
        # Check period_time for string match
        print("\n--- Checking for time '19:25' ---")
        matches = df_blk[df_blk['period_time'].astype(str).str.contains('19:25')]
        if not matches.empty:
            print(matches[existing_cols])
            
            # Print values explicitly
            row = matches.iloc[0]
            if 'team_id' in row and 'home_id' in row:
                print(f"\nRow Detail:")
                print(f"Team ID: {row['team_id']} (Type: {type(row['team_id'])})")
                print(f"Home ID: {row['home_id']} (Type: {type(row['home_id'])})")
                print(f"Away ID: {row['away_id']} (Type: {type(row['away_id'])})")
        else:
             print("No match for 19:25")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
