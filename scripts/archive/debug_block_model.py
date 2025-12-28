import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs, fit_xgs, config

def main():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    clf = joblib.load(model_path)
    block_model = clf.model_block
    feats = clf.config_block.feature_cols
    
    print(f"--- Block Model Analysis ---")
    print(f"Features: {feats}")
    
    # Feature Importance
    importances = block_model.feature_importances_
    f_imp = pd.DataFrame({'feature': feats, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(f_imp)
    
    # Load recent data via project loader
    print("\nLoading data via fit_xgs.load_all_seasons_data()...")
    df = fit_xgs.load_all_seasons_data()
    
    if not df.empty:
        # Filter for 2024 season via game_id
        df['game_id_str'] = df['game_id'].astype(str)
        # 20242025 season games start with 2024...
        mask_2024 = df['game_id_str'].str.startswith('2024')
        # Regular season check (TT=02)
        mask_regular = df['game_id_str'].str[4:6] == '02'
        
        df = df[mask_2024 & mask_regular].copy()
        print(f"Filtered to 2024 Regular Season: {len(df)} rows.")
        
        # Preprocess like the model (includes OHE and filtering)
        df = fit_nested_xgs.preprocess_features(df)
        
        print(f"\nEvaluating on {len(df)} rows...")
        print(f"Model Block Features: {clf.config_block.feature_cols}")
        print(f"DF Columns: {list(df.columns)}")
        
        try:
            p_block = clf.predict_proba_layer(df, 'block')
        except Exception as e:
            print(f"FAILED predict_proba_layer: {e}")
            import traceback
            traceback.print_exc()
            return
            
        auc = roc_auc_score(df['is_blocked'], p_block)
        brier = brier_score_loss(df['is_blocked'], p_block)
        ll = log_loss(df['is_blocked'], p_block)
        
        print(f"AUC:       {auc:.4f}")
        print(f"Brier:     {brier:.4f}")
        print(f"Log Loss:  {ll:.4f}")
        print(f"Mean P:    {p_block.mean():.4f}")
        print(f"Actual Rate: {df['is_blocked'].mean():.4f}")
        
        # Correlation with distance
        if 'distance' in df.columns:
            print(f"Correlation (Prob vs Distance): {pd.Series(p_block).corr(df['distance'].reset_index(drop=True)):.4f}")

if __name__ == "__main__":
    main()
