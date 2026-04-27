
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
    fit_glm, 
    fit_glm_nested, 
    fit_xgboost_nested, 
    fit_xgboost_non_nested, 
    fit_xgboost_alternate,
    analyze,
    config as puck_config
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    modern_seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    
    print(f"============================================================")
    print(f"TRAINING ALL PRODUCTION MODELS (MODERN ERA)")
    print(f"============================================================")
    print(f"Seasons: {modern_seasons}")

    # 1. Load and Aggregate Raw Data
    dfs = []
    for s in modern_seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            df_s = pd.read_csv(csv_path)
            print(f"  [OK] {s}: {len(df_s)} events loaded.")
            dfs.append(df_s)
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
        (fit_glm_nested.NestedGLM, "Nested GLM", "nested_tensor"),
        (fit_glm.NonNestedGLM, "Non-Nested GLM", "non_nested_tensor"),
        (fit_xgboost_nested.XGBNestedXGClassifier, "XGBoost Nested", "xgboost_nested"),
        (fit_xgboost_non_nested.XGBNonNestedXGClassifier, "XGBoost Non-Nested", "xgboost_non_nested"),
        (fit_xgboost_alternate.XGBAlternateXGClassifier, "XGBoost Alternate", "xgboost_alternate")
    ]

    # 3. Train Each Model
    for model_class, name, suffix in models_to_train:
        print(f"\n>>> Training {name}...")
        save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / f"xg_model_{suffix}_modern_era.joblib")
        
        start_t = time.time()
        try:
            # Use the .train() class method which handles preprocessing
            # Note: We pass apply_html_enrichment=False because our CSVs should already be enriched
            # or the pipeline will handle it if needed.
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
    print(f"ALL MODELS TRAINED")
    print(f"============================================================")

if __name__ == "__main__":
    main()
