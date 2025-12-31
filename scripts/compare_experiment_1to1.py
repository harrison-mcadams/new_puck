
import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck import fit_nested_xgs, impute, config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Compare1to1")

def main():
    # 1. Load the Historical Matched Data
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return
    
    df_match_raw = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df_match_raw)} matched shots from comparison file.")
    
    # 2. Train New Model (Balanced + Isotonic) on FULL Dataset (minus this test set? No, just train generic)
    # Ideally we retrain on everything EXCEPT this season to be pure, but for this "Does it fix it?" check, 
    # training on the same distribution is fine.
    
    logger.info("Training New Model (Balanced + Isotonic)...")
    df_train_full = fit_nested_xgs.load_data()
    df_train_full = fit_nested_xgs.preprocess_features(df_train_full)
    df_train_full = impute.impute_blocked_shot_origins(df_train_full, method='point_pull')
    
    # Split for Calibration
    X_train, X_calib = train_test_split(df_train_full, test_size=0.2, random_state=42)
    
    # Train Balanced
    bp = {'class_weight': 'balanced_subsample'}
    clf = fit_nested_xgs.NestedXGClassifier(
        n_estimators=100, max_depth=10, 
        block_params=bp, accuracy_params=bp, finish_params=bp
    )
    clf.fit(X_train)
    
    # Calibrate
    probs_calib = clf.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_calib, (X_calib['event']=='goal').astype(int))
    
    # 3. Predict on Matched Data
    # Preprocess df_match_raw to match model features
    # Map columns if necessary
    # df_match likely has 'distance', 'angle_deg', 'game_state', 'shot_type'.
    
    df_match = df_match_raw.copy()

    # 1. Handle Game State
    if 'game_state' not in df_match.columns:
        if 'situation' in df_match.columns:
            df_match['game_state'] = df_match['situation'] # e.g. '5on5' -> '5v5' might need mapping?
            # Basic mapping
            df_match['game_state'] = df_match['game_state'].replace('5on5', '5v5')
        else:
            # Default to 5v5 if missing (Comparison file is usually 5v5 filtered)
            logger.warning("game_state missing, defaulting to '5v5'")
            df_match['game_state'] = '5v5'

    # 2. Handle Shot Type
    if 'shot_type' not in df_match.columns:
        if 'shotType' in df_match.columns:
            df_match['shot_type'] = df_match['shotType'].str.lower()
        else:
            df_match['shot_type'] = 'unknown'

    # 3. Handle Event (Target)
    # The comparison file target is usually 'is_goal' or similar?
    # We don't strictly need 'event' column for PREDICTION, only for 'preprocess_features' filtering.
    # Since we aren't calling preprocess_features, we are fine.
    
    # Ensure columns exist
    for c in ['distance', 'angle', 'angle_deg']:
        if c in df_match_raw.columns and c not in df_match.columns:
             df_match[c] = df_match_raw[c]
    
    if 'angle_deg' not in df_match.columns and 'angle' in df_match.columns:
        df_match['angle_deg'] = df_match['angle']
             
    # Predict
    # We use clf.predict_proba which MIGHT call OHE internally if columns missing.
    # NestedXGClassifier.predict_proba expects a DataFrame.
    # It will OHE 'game_state', 'shot_type'.
    
    probs_raw = clf.predict_proba(df_match)[:, 1]
    probs_new = iso.predict(probs_raw)
    df_match['xg_new'] = probs_new
    
    # 4. Filter to "Pure Matches" (Location < 1ft, matched features) like before
    # To isolate the *model* difference, not the data difference.
    filters = (df_match['dist_diff'].abs() < 1.0)
    # Check if we have feature matches
    if 'shot_type_match' not in df_match.columns:
         if 'shotType' in df_match.columns:
             df_match['shot_type_match'] = (df_match['shotType'].str.lower() == df_match['shot_type'].str.lower())
             filters &= df_match['shot_type_match']
             
    df_pure = df_match[filters].copy()
    logger.info(f"Analyzing {len(df_pure)} 'Purely Matched' shots (Location & Type match)...")
    
    # 5. Analysis: High Probability Bins
    # We create bins based on MoneyPuck xG (xGoal) to see how we compare
    df_pure['mp_bin'] = pd.cut(df_pure['xGoal'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
    
    logger.info("\n--- Comparison by MoneyPuck Risk Bin ---")
    summary = df_pure.groupby('mp_bin').agg({
        'xgs': 'mean',      # Old Model (from file)
        'xg_new': 'mean',   # New Model
        'xGoal': 'mean',    # MoneyPuck (Target)
        'dist_diff': 'count'
    }).rename(columns={'dist_diff': 'Count', 'xgs': 'My Old', 'xg_new': 'My New', 'xGoal': 'MoneyPuck'})
    
    summary['Old Error'] = summary['My Old'] - summary['MoneyPuck']
    summary['New Error'] = summary['My New'] - summary['MoneyPuck']
    
    print(summary.to_string())
    
    # 6. Global stats
    mae_old = mean_absolute_error(df_pure['xGoal'], df_pure['xgs'])
    mae_new = mean_absolute_error(df_pure['xGoal'], df_pure['xg_new'])
    
    print(f"\nOverall MAE vs MoneyPuck:")
    print(f"Old Model: {mae_old:.4f}")
    print(f"New Model: {mae_new:.4f}")
    
    if mae_new < mae_old:
        print("\nSUCCESS: New model is closer to MoneyPuck!")
    else:
        print("\nNOTE: New model diverged (possibly better? or worse?)")

if __name__ == "__main__":
    main()
