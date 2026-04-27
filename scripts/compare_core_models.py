"""compare_core_models.py

Script to compare the 5 core xG models using a 70-30 train/test split.
Evaluates goal prediction performance using AUC, LogLoss, and Brier Score.

For non-nested models, runs two iterations:
1. Including blocked shots
2. Holding out (excluding) blocked shots
"""

import sys
import os
import pandas as pd
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import config as puck_config
from puck import fit_xgboost_nested
from puck import fit_xgboost_non_nested
from puck import fit_xgboost_alternate
from puck import fit_glm_nested
from puck import fit_glm

def main():
    print("===============================================================")
    print("    CORE MODEL COMPARISON (70-30 SPLIT, MODERN ERA DATA)      ")
    print("===============================================================")

    # 1. Load Modern Era Data
    print("\nLoading Modern Era data (20202021+)...")
    from puck import fit_xgs
    try:
        df_modern = fit_xgs.load_all_seasons_data(min_season=20202021)
        if 'season' in df_modern.columns:
            df_modern = df_modern[df_modern['season'] >= 20202021].copy()
        else:
            df_modern = df_modern[df_modern['game_id'].astype(int) >= 2020000000].copy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(df_modern)} events for comparison.\n")

    # Common testing parameters
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # We will temporarily suppress generating the dashboards by not passing `out_dir`
    # and setting verbose=False for the deep internals to avoid log spam,
    # but the .train methods currently hardcode model_summary calls.
    # We will just let them run and overwrite the default analysis files.

    models_to_test = [
        {
            'name': 'XGBoost Nested',
            'class': fit_xgboost_nested.XGBNestedXGClassifier,
            'kwargs': {'exclude_blocked': False}
        },
        {
            'name': 'XGBoost Alternate (Pure Spatial)',
            'class': fit_xgboost_alternate.XGBAlternateXGClassifier,
            'kwargs': {'exclude_blocked': False}
        },
        {
            'name': 'GLM Nested (Tensor)',
            'class': fit_glm_nested.NestedGLM,
            'kwargs': {'exclude_blocked': False}
        },
        {
            'name': 'XGBoost Non-Nested (Includes Blocks)',
            'class': fit_xgboost_non_nested.XGBNonNestedXGClassifier,
            'kwargs': {'exclude_blocked': False}
        },
        {
            'name': 'XGBoost Non-Nested (Excludes Blocks)',
            'class': fit_xgboost_non_nested.XGBNonNestedXGClassifier,
            'kwargs': {'exclude_blocked': True}
        },
        {
            'name': 'GLM Non-Nested (Includes Blocks)',
            'class': fit_glm.NonNestedGLM,
            'kwargs': {'exclude_blocked': False}
        },
        {
            'name': 'GLM Non-Nested (Excludes Blocks)',
            'class': fit_glm.NonNestedGLM,
            'kwargs': {'exclude_blocked': True}
        }
    ]

    results = []

    for cfg in models_to_test:
        model_name = cfg['name']
        model_class = cfg['class']
        kwargs = cfg['kwargs']
        
        print(f"---------------------------------------------------------------")
        print(f"Training: {model_name}")
        print(f"Config: {kwargs}")
        
        start_t = time.time()
        
        # We add a custom save path to avoid overwriting production models
        # But for this test, we can just save them to a temp prefix
        temp_save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / f'temp_{model_name.replace(" ", "_").lower()}.joblib')
        
        try:
            clf = model_class.train(
                df_modern,
                save_path=temp_save_path,
                verbose=False,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                **kwargs
            )
            
            elapsed = time.time() - start_t
            metrics = getattr(clf, 'test_metrics_', {})
            
            results.append({
                'Model': model_name,
                'AUC': metrics.get('auc', None),
                'LogLoss': metrics.get('logloss', None),
                'Brier': metrics.get('brier', None),
                'Time (s)': f"{elapsed:.1f}"
            })
            print(f"Finished in {elapsed:.1f}s. Metrics: {metrics}")
            
        except Exception as e:
            print(f"FAILED to train {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summarize results
    print("\n===============================================================")
    print("                      COMPARISON RESULTS                       ")
    print("===============================================================")
    
    df_results = pd.DataFrame(results)
    
    # Format float columns
    for col in ['AUC', 'LogLoss', 'Brier']:
        if col in df_results.columns:
            df_results[col] = df_results[col].astype(float).round(5)
            
    # Print as formatted table
    print(df_results.to_string(index=False))
    
    # Save to CSV
    out_csv = Path(puck_config.ANALYSIS_DIR) / 'model_comparison_70_30_results.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

if __name__ == "__main__":
    main()
