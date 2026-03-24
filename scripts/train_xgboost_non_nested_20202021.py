"""train_xgboost_non_nested_20202021.py

Script to train the Non-Nested XGBoost xG model 
specifically on the Modern Era (20202021-present).
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_non_nested, fit_xgs
from puck import config as puck_config

def main():
    print("--- Training Non-Nested XGBoost Model (Modern Era: 20202021+) ---")
    
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    
    print("Loading all seasons data...")
    df_raw = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    
    # 2. Filter for Modern Era
    if 'season' in df_raw.columns:
        df_raw = df_raw[df_raw['season'] >= 20202021].copy()
    else:
        # Fallback to game_id logic if season column is missing
        df_raw = df_raw[df_raw['game_id'].astype(int) >= 2020000000].copy()
        
    print(f"Filtered for seasons 20202021 and onwards. Rows: {len(df_raw)}")
        
    if len(df_raw) == 0:
        print("No rows found for 20202021. Aborting.")
        return

    # 3. Define Save Paths
    save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_non_nested_20202021.joblib')
    
    # 4. Call Consolidated Routine
    # 4. Call Consolidated Routine
    print(f"Fitting model and saving to {save_path}...")
    fit_xgboost_non_nested.XGBNonNestedXGClassifier.train(
        df_raw, 
        save_path=save_path, 
        verbose=True,
        apply_attribution_fix=True,    # Fix defender-owned blocks
        apply_html_enrichment=False,    # Already enriched in CSV
        exclude_blocked=True           # Non-nested models should exclude blocked shots from training target
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model: {save_path}")

if __name__ == "__main__":
    main()
