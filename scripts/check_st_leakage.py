
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path.cwd()))
from puck import fit_xgs, data_pipeline

def check_shot_type_leakage():
    print("Loading data...")
    # Load just one season or a sample to be fast
    df = fit_xgs.load_data() # Defaults to current season or whatever
    
    # Preprocess (needed to get standardized formatting)
    df = data_pipeline.preprocess_features(df, is_training=True, verbose=False)
    
    print(f"Total rows: {len(df)}")
    
    # Check shot_type by event
    if 'shot_type' not in df.columns:
        print("shot_type column missing!")
        return

    events = ['shot-on-goal', 'missed-shot', 'goal', 'blocked-shot']
    df_shots = df[df['event'].isin(events)]
    
    print("\n--- Shot Type Distribution by Event ---")
    print(df_shots.groupby(['event', 'shot_type']).size().unstack(fill_value=0))
    
    # calculate pct unknown
    for evt in events:
        sub = df_shots[df_shots['event'] == evt]
        if len(sub) == 0: continue
        n_unknown = len(sub[sub['shot_type'].isin(['Unknown', 'nan', 'NaN'])])
        print(f"{evt}: {n_unknown} / {len(sub)} Unknown ({n_unknown/len(sub):.1%})")

if __name__ == "__main__":
    check_shot_type_leakage()
