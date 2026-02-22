"""scripts/export_season_shots.py

Exports all shot attempts for the 2025-2026 season to a CSV, including:
- Shot features
- xG (Nested Tensor Model)
- xtG (Mixed Effects Model)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, nhl_api

def main():
    season = "20252026"
    print(f"--- Exporting Season Shots for {season} ---")

    # 1. Load Data
    data_path = f"data/{season}.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # 2. Enrich Team Abbreviations
    # (Adapted from update_mixed_effects.py)
    print("Enriching Team Abbreviations...")
    try:
        # Fetch generic schedule to build ID map
        games = nhl_api.get_season('PHI', season=season)
        id_map = {}
        for g in games:
            if 'awayTeam' in g:
                t = g['awayTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
            if 'homeTeam' in g:
                t = g['homeTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
        
        # Apply Map
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
        df['home_id'] = pd.to_numeric(df['home_id'], errors='coerce')
        df['away_id'] = pd.to_numeric(df['away_id'], errors='coerce')
        
        df['team_abbrev'] = df['team_id'].map(id_map)
        df['home_abb'] = df['home_id'].map(id_map)
        df['away_abb'] = df['away_id'].map(id_map)
        
        # Fallback
        df['team_abbrev'] = df['team_abbrev'].fillna('Unknown')
    except Exception as e:
        print(f"Warning: Failed to enrich team abbrevs: {e}")

    # 3. Preprocess Features (Imputation, Arena Adj, Bio)
    print("Preprocessing features...")
    df = data_pipeline.preprocess_features(
        df,
        is_training=False,
        apply_imputation=True,
        apply_arena_adjustments=True,
        apply_bio_enrichment=True,
        apply_filtering=True # Filters to Goal, Shot, Missed, Blocked
    )
    
    # Explicitly filter for shot attempts if pipeline didn't catch everything or to be safe
    # Pipeline 'apply_filtering' keeps Goal, Shot, Missed Shot, Blocked Shot.
    print(f"Shot attempts after preprocessing: {len(df)}")

    # 4. Load Models
    mixed_model_path = "analysis/xgs/joint_mixed_effects.joblib"
    if not os.path.exists(mixed_model_path):
        print(f"Error: Mixed effects model not found at {mixed_model_path}")
        return

    print(f"Loading Mixed Effects Model from {mixed_model_path}...")
    mixed_model = joblib.load(mixed_model_path)
    
    # Base model is embedded in mixed_model.base_model_
    # But let's verify
    if not hasattr(mixed_model, 'base_model_') or mixed_model.base_model_ is None:
        print("Error: Mixed model does not have a base_model_ attached.")
        return
        
    base_model = mixed_model.base_model_

    # 5. Generate Predictions
    print("Generating predictions...")
    
    # A. Base Model (xG)
    # predict_proba returns [prob_0, prob_1]
    print("  Predicting xG (Nested Tensor)...")
    probs_base = base_model.predict_proba(df)[:, 1]
    df['xg_nested'] = probs_base
    
    # B. Mixed Model (xtG)
    print("  Predicting xtG (Mixed Effects)...")
    probs_mixed = mixed_model.predict_proba(df)[:, 1]
    df['xtg_mixed'] = probs_mixed

    # 6. Select Columns
    output_cols = [
        # Metadata
        'game_id', 'event_id', 'date', 'period', 'period_time', 'team_abbrev', 
        'event', 'goalie_name_x', 'shooter_name', # goalie_name might be goalie_name_x after merges
        
        # Features
        'distance', 'angle_deg', 'x_adj', 'y_adj', 
        'shot_type', 'speed', 'is_rebound', 'is_rush', 
        'time_since_last_event', 'shooter_role', 'shoots_catches', 
        'game_state', 'is_blocked',
        
        # Predictions
        'xg_nested', 'xtg_mixed'
    ]
    
    # Handle column availability
    cols_to_save = [c for c in output_cols if c in df.columns]
    
    # Rename goalie_name_x to goalie_name if present
    if 'goalie_name_x' in cols_to_save:
        df = df.rename(columns={'goalie_name_x': 'goalie_name'})
        cols_to_save = [c if c != 'goalie_name_x' else 'goalie_name' for c in cols_to_save]
        
    df_out = df[cols_to_save]
    
    # 7. Save
    output_path = f"analysis/season_shots_{season}.csv"
    print(f"Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    
    print("Done.")
    print(f"Stats:\n  Count: {len(df_out)}\n  Mean xG: {df_out['xg_nested'].mean():.4f}\n  Mean xtG: {df_out['xtg_mixed'].mean():.4f}")

if __name__ == "__main__":
    main()

