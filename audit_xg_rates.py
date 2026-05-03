
import pandas as pd
import numpy as np
import os
import sys
from puck import timing
from puck import analyze
from puck import data_pipeline

def audit_xg_rates(season='20252026', sample_size=100):
    print(f"Auditing xG rates for season {season} (Sample Size: {sample_size})...")
    
    # 1. Load data
    df_all = timing.load_season_df(season)
    if df_all.empty:
        print("No data found.")
        return

    # 2. Sample games
    all_gids = pd.unique(df_all['game_id'].dropna().astype(int)).tolist()
    if sample_size and len(all_gids) > sample_size:
        import random
        random.seed(42)
        sample_gids = random.sample(all_gids, sample_size)
        df = df_all[df_all['game_id'].isin(sample_gids)].copy()
        print(f"Sampled {sample_size} games out of {len(all_gids)}.")
    else:
        df = df_all
        print(f"Using all {len(all_gids)} games.")

    # 3. Define condition
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # 4. Use xgs_map to get filtered data and stats
    print("Running xgs_map to get filtered data and stats...")
    res = analyze.xgs_map(
        season=season,
        data_df=df,
        condition=condition,
        heatmap_only=True,
        return_filtered_df=True,
        normalize_per60=True,
        show=False,
        return_heatmaps=True
    )
    
    out_path, heatmaps, df_filtered, stats = res

    if df_filtered.empty:
        print("No 5v5 events found in sample.")
        return

    # 5. Extract Overall Stats
    total_xg = stats.get('team_xgs', 0) + stats.get('other_xgs', 0)
    total_goals = stats.get('team_goals', 0) + stats.get('other_goals', 0)
    total_seconds = stats.get('team_seconds', 0)
    
    xg_per_60 = (total_xg / total_seconds) * 3600 if total_seconds > 0 else 0
    goals_per_60 = (total_goals / total_seconds) * 3600 if total_seconds > 0 else 0
    
    print(f"\n--- 5v5 Sample Totals ---")
    print(f"Games Sampled: {len(pd.unique(df_filtered['game_id']))}")
    print(f"Total Seconds: {total_seconds:.2f} ({total_seconds/3600:.2f} hours)")
    print(f"Total xG:      {total_xg:.2f}")
    print(f"Total Goals:   {total_goals}")
    print(f"League xG/60:  {xg_per_60:.3f}")
    print(f"League G/60:   {goals_per_60:.3f}")
    print(f"G/xG Ratio:    {total_goals/total_xg:.3f}" if total_xg > 0 else "N/A")

    # 6. Rebound Potency Audit
    print(f"\n--- Rebound Potency Audit (5v5 Sample) ---")
    attempt_types = {'goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'}
    df_shots = df_filtered[df_filtered['event'].astype(str).str.strip().str.lower().isin(attempt_types)]
    
    df_reb = df_shots[df_shots['is_rebound'] == 1]
    df_non_reb = df_shots[df_shots['is_rebound'] == 0]
    
    def get_group_stats(group_df):
        if group_df.empty: return None
        g_total_xg = group_df['xgs'].sum()
        g_goals = (group_df['event'].astype(str).str.strip().str.lower() == 'goal').sum()
        g_count = len(group_df)
        return {
            'count': g_count,
            'avg_xg': g_total_xg/g_count,
            'goal_rate': g_goals/g_count,
            'g_xg_ratio': g_goals/g_total_xg if g_total_xg > 0 else 0
        }

    s_reb = get_group_stats(df_reb)
    s_non = get_group_stats(df_non_reb)
    
    if s_reb:
        print(f"Rebounds: Count={s_reb['count']}, Avg xG={s_reb['avg_xg']:.4f}, Goal Rate={s_reb['goal_rate']:.4f}, G/xG={s_reb['g_xg_ratio']:.3f}")
    if s_non:
        print(f"Non-Rebounds: Count={s_non['count']}, Avg xG={s_non['avg_xg']:.4f}, Goal Rate={s_non['goal_rate']:.4f}, G/xG={s_non['g_xg_ratio']:.3f}")

    # Breakdown rebounds by last event
    print("\nRebounds by Last Event Type:")
    if not df_reb.empty:
        for let in df_reb['last_event_type'].unique():
            df_let = df_reb[df_reb['last_event_type'] == let]
            s_let = get_group_stats(df_let)
            if s_let:
                print(f"  {str(let):15}: Count={s_let['count']:4}, Avg xG={s_let['avg_xg']:.4f}, Goal Rate={s_let['goal_rate']:.4f}, G/xG={s_let['g_xg_ratio']:.3f}")

    # 7. Model Feature Importance
    print(f"\n--- Model Feature Importance ---")
    try:
        # Default model used by analyze.py
        model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_tensor_modern_era.joblib')
        if os.path.exists(model_path):
            import joblib
            model = joblib.load(model_path)
            print(f"Loaded model: {type(model).__name__}")
            
            features = model.features
            for label, attr in [('Block', 'model_block'), ('Accuracy', 'model_acc'), ('Finish', 'model_finish')]:
                clf = getattr(model, attr, None)
                if not clf: continue
                
                # Handle Pipeline (NestedGLM)
                if hasattr(clf, 'named_steps'):
                    final_est = clf.named_steps.get('clf')
                    if hasattr(final_est, 'coef_'):
                        print(f"\n{label} Layer (GLM) Coefficients:")
                        # coefficients are harder to map to features due to preprocessing (Splines/OHE)
                        print("  (GLM coefficients not easily mapped to raw features in this audit)")
                        continue
                    clf = final_est # Fallback to check for importances in pipeline step
                
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    # Determine feature names (XGBTensorXGClassifier has features_block, etc.)
                    f_attr = f'features_{label.lower()[:3]}' if label != 'Accuracy' else 'features_acc'
                    current_features = getattr(model, f_attr, features)
                    
                    if len(importances) != len(current_features):
                        print(f"  Warning: importance length mismatch ({len(importances)} vs {len(current_features)})")
                        feat_imp = sorted(zip([f"f{i}" for i in range(len(importances))], importances), key=lambda x: x[1], reverse=True)
                    else:
                        feat_imp = sorted(zip(current_features, importances), key=lambda x: x[1], reverse=True)
                    
                    print(f"\n{label} Layer Top 10 Features:")
                    for f, imp in feat_imp[:10]:
                        print(f"  {f:25}: {imp:.4f}")
                    
                    reb_feats = ['is_rebound', 'rebound_angle_change', 'rebound_time_diff']
                    print(f"  {label} Rebound Features Importance:")
                    for f, imp in feat_imp:
                        if any(rf in f for rf in reb_feats):
                            print(f"    {f:25}: {imp:.4f}")
    except Exception as e:
        print(f"Could not load feature importances: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else '20252026'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    audit_xg_rates(season, n)
