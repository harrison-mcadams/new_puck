
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs, analyze, config, fit_xgs

def debug_features():
    print("Loading 20252026 data...")
    df = fit_xgs.load_all_seasons_data(config.DATA_DIR)
    
    # Filter 5v5
    mask_5v5 = (df['game_state'] == '5v5') & (df['is_net_empty'] == 0)
    df_5v5 = df[mask_5v5].copy()
    print(f"5v5 Data Shape: {df_5v5.shape}")
    
    # Load Model (Nested_All)
    model_path = os.path.join(config.ANALYSIS_DIR, 'xgs', 'xg_model_nested_all.joblib')
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    print("\nModel Internal Config:")
    if hasattr(clf, 'features'):
        print(f"Features: {clf.features}")
    if hasattr(clf, 'categorical_cols'):
        print(f"Categorical Cols: {clf.categorical_cols}")
        
    # Mimic analyze.py preparation
    print("\nPreparing Data (mimicking analyze.py)...")
    
    # 1. Impute (as in analyze._predict_xgs)
    try:
        from puck import impute
        df_imputed = impute.impute_blocked_shot_origins(df_5v5, method='mean_6')
    except Exception as e:
        print(f"Impute fallback: {e}")
        df_imputed = df_5v5.copy()
        
    if 'shot_type' in df_imputed.columns:
        df_imputed['shot_type'] = df_imputed['shot_type'].fillna('Unknown')
    else:
        df_imputed['shot_type'] = 'Unknown'
        
    # Enrich
    df_imputed = fit_xgs.enrich_data_with_bios(df_imputed)
    
    # Clean
    input_features = getattr(clf, 'features', ['distance', 'angle_deg', 'game_state', 'shot_type'])
    # fit_xgs.clean_df_for_model calls fit_xgs.one_hot_encode if encode_method != 'none'.
    # analyze.py calls it with encode_method='none'.
    
    df_model, final_features, cat_map = fit_xgs.clean_df_for_model(
        df_imputed, input_features, encode_method='none'
    )
    
    print(f"\nPrepared DataFrame Head (Columns: {df_model.columns.tolist()}):")
    print(df_model.head())
    
    print("\nCategorical Column Values in Prepared DF:")
    cat_cols = ['game_state', 'shot_type', 'shoots_catches']
    for c in cat_cols:
        if c in df_model.columns:
            print(f"Unique {c}: {df_model[c].unique()}")
        else:
            print(f"{c} MISSING from df_model!")
            
    # Now check what happens inside predict_proba
    # We can't step into it, but we can verify the get_dummies result
    print("\nSimulating Internal OHE (as in NestedXGClassifier.predict_proba):")
    df_internal = df_model.copy()
    
    if hasattr(clf, 'categorical_cols') and clf.categorical_cols:
        print(f"Model expects categorical cols: {clf.categorical_cols}")
        # NestedXGClassifier logic:
        # if cat_cols and not any(c in df.columns for c in self.final_features if c not in self.features):
        #      df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_')
        
        df_encoded = pd.get_dummies(df_internal, columns=clf.categorical_cols, prefix_sep='_')
        print(f"Encoded Columns ({len(df_encoded.columns)}): {df_encoded.columns.tolist()}")
        
        # Check specific expected features
        if hasattr(clf, 'final_features'):
            missing = [f for f in clf.final_features if f not in df_encoded.columns]
            print(f"Missing Final Features ({len(missing)}): {missing}")
            
            # Check interaction with 5v5
            gs_5v5 = 'game_state_5v5'
            if gs_5v5 in df_encoded.columns:
                print(f"'{gs_5v5}' exists. Mean: {df_encoded[gs_5v5].mean()}")
            else:
                print(f"CRITICAL: '{gs_5v5}' does NOT exist!")
    
if __name__ == "__main__":
    debug_features()
