
import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import config, fit_nested_xgs, fit_xgboost_nested, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("VerifyTotals")

def main():
    logger.info("Starting Global Totals Verification...")
    
    # 1. Load Model
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested.joblib'
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {type(model).__name__}")
    
    # 2. Load Data
    logger.info("Loading all available season data...")
    df = fit_nested_xgs.load_data()
    
    # Filter regular season based on game_id
    if 'game_id' in df.columns:
        df['game_id_str'] = df['game_id'].astype(str)
        df = df[df['game_id_str'].str.contains(r'^\d{4}02\d{4}$')]
    else:
        logger.warning("No 'game_id' found, processing full dataset (might include pre-season).")
    
    df = fit_xgboost_nested.preprocess_data(df)
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # CRITICAL: Filter out Period 5 (Shootouts) to align universes
    df = df[df['period_number'] <= 4].copy()
    
    logger.info(f"Processing {len(df)} events...")
    
    # 3. Predict
    probs = model.predict_proba(df)[:, 1]
    df['xG'] = probs
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    
    # 4. Aggregate
    total_xg = df['xG'].sum()
    total_goals = df['is_goal'].sum()
    ratio = total_xg / total_goals if total_goals > 0 else 0
    
    print("\n" + "="*40)
    print(f"GLOBAL REGULAR SEASON SUMMARY")
    print("="*40)
    print(f"Total Shots Analyzed: {len(df):,}")
    print(f"Total Actual Goals:   {total_goals:,}")
    print(f"Total Expected Goals: {total_xg:,.2f}")
    print(f"xG / Goals Ratio:     {ratio:.4f}")
    print("="*40)

    # Event Breakdown
    print("\nBreakdown by Event Type:")
    breakdown = df.groupby('event').agg({
        'xG': ['count', 'sum', 'mean'],
        'is_goal': 'sum'
    })
    print(breakdown.to_string())

    unblocked = df[df['event'] != 'blocked-shot']
    u_xg = unblocked['xG'].sum()
    u_goals = unblocked['is_goal'].sum()
    print(f"\nUnblocked xG/Goal Ratio: {u_xg/u_goals:.4f}")
    
    if abs(1-ratio) < 0.05:
        print("SUCCESS: Model is globally well-calibrated (within 5% error).")
    else:
        print("WARNING: Model shows global bias. Calibration might need re-tuning.")

if __name__ == "__main__":
    main()
