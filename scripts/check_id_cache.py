
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from puck import fit_xgs

def check_id_mapping():
    print("Checking ID mapping...")
    try:
        # Load batch summary
        df_batch = pd.read_csv('analysis/blocked_shots/blocked_shots_summary_batch.csv')
        sample = df_batch.iloc[0]
        game_id = sample['game_id']
        block_id = sample['block_id']
        
        print(f"Sample: Game {game_id}, Block ID {block_id}")
        
        # Load PBP
        # Assume standard location
        pbp_path = f'data/{str(game_id)[:4]}{str(int(str(game_id)[:4])+1)}/{game_id}.csv' # This assumes naming convention maybe?
        # Better: use load_data or finding the file.
        # fit_xgs.load_data() loads ALL. Slow.
        # Let's try finding the file directly.
        season = str(game_id)[:4] + str(int(str(game_id)[:4])+1)
        pbp_path = f'data/{season}/{season}_df.csv'
        
        if os.path.exists(pbp_path):
            print(f"Loading {pbp_path}...")
            df_pbp = pd.read_csv(pbp_path)
            # Filter
            event = df_pbp[(df_pbp['game_id'] == game_id) & (df_pbp['event_id'] == block_id)]
            if not event.empty:
                print("Found matching event!")
                print(event[['event', 'event_id', 'period', 'period_time_desc', 'x', 'y']])
                if event.iloc[0]['event'] == 'blocked-shot':
                     print("SUCCESS: ID maps to 'blocked-shot' event.")
                else:
                     print(f"WARNING: ID maps to '{event.iloc[0]['event']}'")
            else:
                print("Event ID not found in PBP.")
        else:
            print(f"PBP file {pbp_path} not found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_id_mapping()
