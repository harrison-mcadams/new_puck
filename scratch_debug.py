import pandas as pd
from puck import analyze

try:
    df_filtered = analyze.xgs_map(game_id='2025030161', return_heatmaps=False, return_filtered_df=True)[2]
    
    for mn in ['analysis/xgs/xg_model_xgboost_nested.joblib', 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib']:
        df_with_xgs, clf, clf_meta = analyze._predict_xgs(
            df_filtered,
            model_path=mn,
            behavior='load',
            csv_path=None,
            preprocess=False
        )
        
        valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
        valid_df = df_with_xgs[df_with_xgs['event'].isin(valid_events)]
        
        print("\nModel:", mn)
        print(valid_df.groupby('event')['xgs'].describe())
        
except Exception as e:
    print(f"Exception: {e}")
