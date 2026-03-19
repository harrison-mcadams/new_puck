"""train_nested_glm_20202021.py

Script to train the Nested Polynomial Logistic Regression (GLM) xG model 
specifically on the 2020-2021 season data.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_glm_nested, fit_xgs
from puck import config as puck_config

def main():
    print("--- Training Nested GLM (Poly/Tensor) Model (20202021 Season ONLY) ---")
    
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / 'data' / '20202021' / '20202021_df.csv'
    
    if not data_file.exists():
        print(f"Error: 20202021 data file not found at {data_file}")
        # Try fallback load_all_seasons and filter
        print("Attempting to load all seasons and filter for 20202021...")
        df_raw = fit_xgs.load_all_seasons_data()
        if 'season' in df_raw.columns:
            df_raw = df_raw[df_raw['season'] == 20202021].copy()
        else:
            # Maybe it's in game_id?
            # 2020020001
            df_raw = df_raw[df_raw['game_id'].astype(str).str.startswith('2020')].copy()
    else:
        print(f"Loading data from {data_file}...")
        df_raw = pd.read_csv(data_file)
        
    print(f"Loaded {len(df_raw)} rows.")
    if len(df_raw) == 0:
        print("No rows found for 20202021. Aborting.")
        return

    # 2. Define Save Paths
    save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested_tensor_20202021.joblib')
    out_dir = Path(puck_config.ANALYSIS_DIR) / 'nested_xgs_20202021'
    
    # 3. Call Consolidated Routine
    print(f"Fitting model and saving to {save_path}...")
    fit_glm_nested.train_nested_glm(df_raw, save_path=save_path, verbose=True)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model: {save_path}")
    print(f"Diagnostics: {out_dir}")

if __name__ == "__main__":
    main()
