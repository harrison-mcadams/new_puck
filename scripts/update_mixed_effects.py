"""scripts/update_mixed_effects.py

Training script for v2 Mixed Effects Model.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects
from puck import config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Mixed Effects Model")
    parser.add_argument("--teams", type=str, help="Comma-separated list of teams to generate plots for (e.g. PHI,PIT)", default=None)
    parser.add_argument("--skip-training", action="store_true", help="Skip training and just regenerate plots if model exists")
    args = parser.parse_args()

    season = "20252026"
    print(f"--- Training Mixed Effects v2 for {season} ---")
    
    # 1. Load Data
    data_path = "data/20252026.csv"
    if not os.path.exists(data_path):
        # Fallback to test data or fetch?
        # For now, assume it exists or fail
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # --- 2b. Enrich Team Abbreviations ---
    # The raw CSV has team_id but not team_abbrev. We need abbrevs for the model/plots.
    print("Enriching Team Abbreviations...")
    try:
        from puck import nhl_api
        # Fetch appropriate season schedule to get team mapping
        # We can just fetch 'PHI' schedule for 20252026 to see all opponents + themselves
        # Actually PHI plays everyone, so we should get all active teams.
        games = nhl_api.get_season('PHI', season='20252026')
        
        id_map = {}
        for g in games:
            # Check for away/home team blocks
            # Structure usually: g['awayTeam']['id'], g['awayTeam']['abbrev']
            if 'awayTeam' in g:
                t = g['awayTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
                    
            if 'homeTeam' in g:
                t = g['homeTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
                    
        print(f"Built ID Map for {len(id_map)} teams: {sorted(id_map.values())}")
        
        # Apply Map
        # Ensure IDs are ints for mapping
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
        df['home_id'] = pd.to_numeric(df['home_id'], errors='coerce')
        df['away_id'] = pd.to_numeric(df['away_id'], errors='coerce')
        
        df['team_abbrev'] = df['team_id'].map(id_map)
        df['home_abb'] = df['home_id'].map(id_map)
        df['away_abb'] = df['away_id'].map(id_map)
        
        # Fallback for unmapped (shouldn't happen for active teams)
        df['team_abbrev'] = df['team_abbrev'].fillna('Unknown')
        df['home_abb'] = df['home_abb'].fillna('Unknown')
        df['away_abb'] = df['away_abb'].fillna('Unknown')
        
    except Exception as e:
        print(f"Warning: Failed to enrich team abbrevs: {e}")
        # data_pipeline will fill 'Unknown'
    
    # Filter for standard shot events (though pipeline checks too)
    target_events = ['Goal', 'Shot', 'Missed Shot', 'Blocked Shot', 'goal', 'shot', 'missed-shot', 'blocked-shot', 'shot-on-goal']
    df = df[df['event'].isin(target_events)]
    
    logger.info(f"Training on {len(df)} shot events.")
    logger.info(f"Raw Event Counts (Pre-Pipeline):\n{df['event'].value_counts()}")

    # 1.5 Preprocess & Impute
    # The Base Model expects clean data with imputed coordinates for blocked shots
    # We use the centralized pipeline which handles:
    # - Orientation & Arena Adjustments
    # - Imputation of Blocked Shot Origins
    # - Bio Enrichment (Shoots/Catches)
    # - Feature Formatting (Filling Defaults)
    try:
        from puck import data_pipeline
    except ImportError:
        print("Importing data_pipeline failed")
        return

    print("Preprocessing using robust pipeline (imputation, arena adj, bio enrichment)...")
    df = data_pipeline.preprocess_features(
        df,
        is_training=False,     # We don't need dithering for ME model context
        apply_imputation=True, # Critical: Impute blocked shot coords
        apply_arena_adjustments=True, # Critical: Adjust for arena bias
        apply_bio_enrichment=True,    # Critical: Add shoots_catches/shooter_role
        apply_filtering=True   # Filter to valid shot events and remove empty net
    )
    
    # Ensure target 'is_goal' exists (mixed_effects model usually expects 'y' or calculates it from 'event')
    # The pipeline cleans features but might not create 'is_goal' if not training flag on?
    # Actually data_pipeline returns df, doesn't add is_goal unless specified.
    if 'is_goal' not in df.columns:
        df['is_goal'] = (df['event'] == 'goal').astype(int)
        
    logger.info(f"Cleaned data size: {len(df)}")
    
    logger.info("--- DEBUG: Target Variable Check ---")
    if 'event' in df.columns:
        # manual formatting for log
        counts = df['event'].value_counts()
        logger.info(f"Event Counts:\n{counts}")
    if 'is_goal' in df.columns:
        logger.info(f"is_goal Sum: {df['is_goal'].sum()}")
        logger.info(f"is_goal Mean: {df['is_goal'].mean():.4f}")
    logger.info("------------------------------------")
    
    # Check for NaNs just in case
    if 'distance' in df.columns:
        n_nan = df['distance'].isna().sum()
        if n_nan > 0:
            logger.info(f"Warning: {n_nan} rows have NaN distance after pipeline. Filling with 0.")
            df['distance'] = df['distance'].fillna(0)
            
    # 2. Init Model
    out_path = "analysis/xgs/joint_mixed_effects.joblib"
    
    if args.skip_training and os.path.exists(out_path):
        print(f"Skipping training, loading model from {out_path}...")
        mixed = joblib.load(out_path)
    else:
        print("Initializing Mixed Effects Model...")
        # We use the Nested Tensor/GLM model as base
        # Enable Tensor Splines for the random effects too
        mixed = mixed_effects.GameMixedEffectsXG(
            base_model_path="analysis/xgs/xg_model_nested_tensor.joblib",
            feature_set=[], # No features needed for random intercepts
            use_tensor_splines=False, # Disable expensive tensor splines
            component_model_type='intercept', # Learn only team intercepts
            l2_reg=1.0 # Standard regularization for intercepts
        )
        
        # 3. Fit
        print("Fitting model...")
        # fit() handles base margin prediction internally
        mixed.fit(df)
        
        # 4. Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(mixed, out_path)
        print(f"Saved model to {out_path}")
    
    # 5. Diagnostics & Saving
    print("\n--- Saving Summary & Generating Plots ---")
    summary_dir = "analysis/xgs/mixed_effects"
    
    # Parse teams filter
    teams_filter = None
    if args.teams:
        teams_filter = args.teams.split(',')
        print(f"Filtering plots for teams: {teams_filter}")

    mixed.save_summary(summary_dir, teams_filter=teams_filter)
    print(f"Summary saved to {summary_dir}")

    # 6. Enhanced Visualization (Relative Maps & Scatter)
    # Only run this full league viz if no filter, OR modify it to support filter too
    # For now, skip if filtering specific teams to be fast
    # 6. Enhanced Visualization (Relative Maps & Scatter)
    # Modified to support filtering or full run
    print("Generating Spatial Maps (Heatmaps)...")
    
    # Needs Predictions in DF for summary stats?
    # Yes, we want to show actual Goal/xG counts.
    # We must run predict_proba to get 'xg_mixed' column.
    if 'xg_mixed' not in df.columns:
        print("Generating predictions for summary stats...")
        try:
             # Ensure df has necessary cols (pipeline should have handled it)
             # df should be same as training data
             probs = mixed.predict_proba(df)
             df['xg_mixed'] = probs
        except Exception as e:
             print(f"Warning: Could not generate predictions for stats: {e}")
             df['xg_mixed'] = 0.0

    try:
        from puck import mixed_effects_viz
        # Pass teams_filter (list or None) AND df (for stats)
        mixed_effects_viz.generate_spatial_grids(mixed, output_dir="analysis/xgs/mixed_effects/viz", teams=teams_filter, df=df)
    except Exception as e:
        print(f"Failed to generate enhanced viz: {e}")
        import traceback
        traceback.print_exc()

    print("Update Complete.")
if __name__ == "__main__":
    main()
