
import pandas as pd
import parse
import sys
import os

# Mock timing import
try:
    import timing_new as timing
    sys.modules['timing'] = timing
except ImportError:
    import timing

def run_debug():
    print("--- Debugging League Filtering ---")
    
    # Load season DF
    season = '20252026'
    try:
        df = timing.load_season_df(season)
    except AttributeError:
        print("timing.load_season_df not found!")
        return

    if df is None or df.empty:
        print("Season DF is empty or None")
        return

    print(f"Loaded DF shape: {df.shape}")
    if 'game_state' in df.columns:
        print("game_state column present")
        print(f"Unique game_states: {df['game_state'].unique()}")
        print(f"Null game_states: {df['game_state'].isna().sum()}")
    else:
        print("game_state column MISSING")

    # Try filtering
    condition = {'game_state': ['5v5']}
    print(f"\nApplying condition: {condition}")
    
    try:
        mask = parse.build_mask(df, condition)
        print(f"Mask shape: {mask.shape}, True count: {mask.sum()}")
        
        # Simulate analyze.py logic
        mask = mask.reindex(df.index).fillna(False).astype(bool)
        df_cond = df.loc[mask].copy()
        print(f"Filtered DF shape: {df_cond.shape}")
        
        if len(df_cond) == len(df):
            print("WARNING: Filtered DF has same length as original (Filtering might be ineffective)")
        elif len(df_cond) == 0:
            print("WARNING: Filtered DF is empty")
        else:
            print("Filtering seems to work (subset returned)")
            
    except Exception as e:
        print(f"FAILURE: build_mask raised exception: {e}")

if __name__ == '__main__':
    run_debug()
