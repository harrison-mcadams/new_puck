import pandas as pd
import os
import glob

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

def inspect():
    if not os.path.exists(METADATA_FILE):
        print("Metadata not found")
        return

    df_meta = pd.read_csv(METADATA_FILE)
    print("Metadata Sample:")
    print(df_meta[['game_id', 'event_id', 'time', 'period']].head(3))
    
    # Grab a sample file
    files = glob.glob(os.path.join(DATA_DIR, "*positions.csv"))
    if not files:
        print("No position files found")
        return
        
    fpath = files[0]
    print(f"\nInspecting: {os.path.basename(fpath)}")
    df_pos = pd.read_csv(fpath)
    
    print("\nPositions Sample:")
    print(df_pos[['frame_idx', 'timestamp', 'entity_type']].head(3))
    
    print("\nTimestamp Range:")
    print(f"Start: {df_pos['timestamp'].min()}")
    print(f"End:   {df_pos['timestamp'].max()}")
    
    # Check if we can parse the timestamp
    # Often it's YYYY-MM-DD HH:MM:SS format
    try:
        sample_ts = df_pos['timestamp'].iloc[0]
        # It references absolute time. Metadata 'time' is game clock 'MM:SS'.
        # We assume they are unrelated unless we know the period start time.
    except:
        pass

if __name__ == "__main__":
    inspect()
