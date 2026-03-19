"""train_glm_20202021.py

Script to train the Non-Nested Polynomial Logistic Regression (GLM) xG model 
specifically on the 2020-2021 through 2025-2026 seasons.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_glm, fit_xgs
from puck import config as puck_config

def main():
    print("--- Training Non-Nested GLM (Poly/Tensor) Model (Modern Era: 20202021+) ---")
    
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    
    print("Loading all seasons data...")
    df_raw = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    
    # 2. Filter for Modern Era
    # Looking for 'season' column
    if 'season' in df_raw.columns:
        df_raw = df_raw[df_raw['season'] >= 20202021].copy()
    else:
        # Fallback to game_id logic if season column is missing
        # game_id 2020020001
        df_raw = df_raw[df_raw['game_id'].astype(int) >= 2020000000].copy()
        
    print(f"Filtered for seasons 20202021 and onwards. Rows: {len(df_raw)}")
    
    if len(df_raw) == 0:
        print("No rows found for modern era. Aborting.")
        return

    # 3. Define Save Path
    save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_non_nested_tensor_20202021.joblib')
    
    # 4. Call Consolidated Routine
    print(f"Fitting model and saving to {save_path}...")
    fit_glm.NonNestedGLM.train(df_raw, save_path=save_path, verbose=True)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model: {save_path}")

if __name__ == "__main__":
    main()
