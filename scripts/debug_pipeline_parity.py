import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, analyze, features as feature_util

def debug_pipeline():
    print("============================================================", flush=True)
    print("DATA PIPELINE DIAGNOSTIC: Parity Check", flush=True)
    print("============================================================", flush=True)

    season = '20252026'
    try:
        csv_path = analyze.locate_season_csv(season)
        print(f"Loading from: {csv_path}", flush=True)
        df_raw = pd.read_csv(csv_path)
        print(f"Loaded {season}: {len(df_raw)} events.", flush=True)
    except Exception as e:
        print(f"Failed to load {season}: {e}", flush=True)
        return

    # Sample to speed up
    if len(df_raw) > 10000:
        df_raw = df_raw.sample(10000, random_state=42)
        print(f"Sampled to 10,000 events.", flush=True)

    print("\nRunning data_pipeline.preprocess_features (NO HTML, WITH FILTERING)...", flush=True)
    df = data_pipeline.preprocess_features(
        df_raw, 
        is_training=True, 
        verbose=True,
        apply_html_enrichment=False,
        apply_filtering=True
    )

    feature_list = feature_util.get_features('all_inclusive')
    
    print("\n--- Feature Presence Check ---", flush=True)
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"MISSING FEATURES: {missing}", flush=True)
    else:
        print("All features present.", flush=True)

    print("\n--- Feature Statistics (Full Sample) ---", flush=True)
    numeric_feats = df[feature_list].select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numeric_feats].describe().T[['mean', 'std', 'min', 'max']]
    # Add NaN count
    stats['nan_count'] = df[numeric_feats].isna().sum()
    print(stats)

    print("\n--- Categorical Feature Distributions ---", flush=True)
    cat_feats = [f for f in feature_list if f not in numeric_feats]
    for f in cat_feats:
        print(f"\n{f}:", flush=True)
        print(df[f].value_counts(normalize=True).head(5))

    print("\n--- Block Model Feature Parity Check ---", flush=True)
    # Features that were previously held out
    held_out = [
        'shot_type', 
        'dist_from_last_event', 
        'speed_from_last_event', 
        'angle_change_last_event',
        'rebound_angle_change',
        'rebound_time_diff',
        'rebound_source',
        'is_rebound',
        'last_event_type',
        'last_event_time_diff'
    ]
    
    print("\nComparing Blocked vs Unblocked for held-out features:", flush=True)
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    for f in held_out:
        if f in df.columns:
            if df[f].dtype.kind in 'iufc':
                # Numeric
                means = df.groupby('is_blocked')[f].mean()
                print(f"{f} mean -> Unblocked: {means.get(0, 0):.4f}, Blocked: {means.get(1, 0):.4f}", flush=True)
            else:
                # Categorical
                top_blocked = df[df['is_blocked'] == 1][f].value_counts(normalize=True).head(3).to_dict()
                top_unblocked = df[df['is_blocked'] == 0][f].value_counts(normalize=True).head(3).to_dict()
                print(f"{f} top -> Unblocked: {top_unblocked}, Blocked: {top_blocked}", flush=True)

    print("\n--- Sensible Value Check ---", flush=True)
    # Check for extreme distance/angles
    if 'distance' in df.columns:
        extreme_dist = df[df['distance'] > 100]
        if len(extreme_dist) > 0:
            print(f"WARNING: {len(extreme_dist)} shots with distance > 100ft.", flush=True)
    
    if 'x' in df.columns:
        # We expect x to be mostly > 0 after standardization (attacking right)
        x_neg = (df['x'] < 0).mean()
        print(f"Standardized X < 0: {x_neg:.1%} (Expected to be small for shots)", flush=True)

    print("\n============================================================", flush=True)

if __name__ == "__main__":
    debug_pipeline()
