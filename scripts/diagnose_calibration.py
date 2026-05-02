import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline

def check_calibration():
    # Monkeypatch to skip slow enrichment
    original_preprocess = data_pipeline.preprocess_features
    def fast_preprocess(*args, **kwargs):
        kwargs['apply_html_enrichment'] = False
        return original_preprocess(*args, **kwargs)
    data_pipeline.preprocess_features = fast_preprocess
    
    model_path = 'analysis/xgs/xg_model_xgboost_tensor_modern_era.joblib'
    print(f"Checking model: {model_path}")
    
    seasons = ['20232024', '20242025', '20252026']
    results = []
    
    for s in seasons:
        try:
            path = analyze.locate_season_csv(s)
            df = pd.read_csv(path)
            # Filter for shot events (to match training)
            shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
            df = df[df['event'].isin(shot_events)].copy()
            
            df, _, _ = analyze._predict_xgs(df, model_path=model_path)
            
            g = df[df['event'] == 'goal'].shape[0]
            xg = df['xgs'].sum()
            ratio = g / xg if xg > 0 else 0
            
            results.append({
                'season': s,
                'shots': len(df),
                'goals': g,
                'xg': round(xg, 2),
                'ratio': round(ratio, 3)
            })
            print(f"  {s}: Ratio={ratio:.3f}")
        except Exception as e:
            print(f"  {s}: Error - {e}")
            
    df_res = pd.DataFrame(results)
    print("\nCalibration Summary:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    check_calibration()
