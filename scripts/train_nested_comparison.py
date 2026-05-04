
import os
import sys
import pandas as pd
import joblib
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import (
    fit_glm_nested, 
    fit_xgboost_nested, 
    fit_xgboost_tensor,
    analyze,
    config as puck_config
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    modern_seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    
    print(f"============================================================")
    print(f"TRAINING NESTED COMPARISON MODELS (MODERN ERA)")
    print(f"============================================================")
    print(f"Seasons: {modern_seasons}")

    # 1. Load and Aggregate Raw Data
    dfs = []
    for s in modern_seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            if csv_path:
                df_s = pd.read_csv(csv_path, low_memory=False)
                print(f"  [OK] {s}: {len(df_s)} events loaded.")
                dfs.append(df_s)
            else:
                print(f"  [MISSING] {s}: CSV not found.")
        except Exception as e:
            print(f"  [Error] Failed to load {s}: {e}")

    if not dfs:
        print("Error: No data loaded.")
        return

    df_modern = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Modern Era Dataset: {len(df_modern)} events.")

    # 2. Define Models to Train
    # Format: (Class, Name, SaveSuffix)
    models_to_train = [
        (fit_xgboost_tensor.XGBTensorXGClassifier, "XGBoost Tensor", "xgboost_tensor_modern_era"),
        (fit_xgboost_nested.XGBNestedXGClassifier, "XGBoost Nested (XGB+GLM)", "xgboost_nested_modern_era"),
        (fit_glm_nested.NestedGLM, "Nested GLM (Standard)", "nested_tensor_modern_era")
    ]

    # 3. Train Each Model
    for model_class, name, suffix in models_to_train:
        print(f"\n>>> Training {name}...")
        save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / f"xg_model_{suffix}.joblib")
        
        start_t = time.time()
        try:
            # Use the .train() class method which handles preprocessing
            model_class.train(
                df_modern, 
                save_path=save_path,
                verbose=True,
                apply_html_enrichment=False
            )
            print(f"  [SUCCESS] {name} saved to {save_path}")
            print(f"  Time: {time.time() - start_t:.1f}s")
        except Exception as e:
            print(f"  [FAILURE] {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n============================================================")
    print(f"NESTED COMPARISON TRAINING COMPLETE")
    print(f"============================================================")

if __name__ == "__main__":
    main()
