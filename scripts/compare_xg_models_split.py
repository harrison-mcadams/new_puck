
import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
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
    data_pipeline,
    features as feature_util
)

def evaluate_submodels(model, df_test):
    """Evaluates nested submodels (Block, Accuracy, Finish)."""
    results = {}
    
    # 1. Block Layer (Target: is_blocked)
    y_block = (df_test['event'] == 'blocked-shot').astype(int)
    p_block = np.clip(model.predict_proba_layer(df_test, 'block'), 0, 1)
    results['block'] = {
        'auc': roc_auc_score(y_block, p_block),
        'logloss': log_loss(y_block, p_block),
        'brier': brier_score_loss(y_block, p_block)
    }
    
    # 2. Accuracy Layer (Target: on_net | unblocked)
    df_unblocked = df_test[df_test['event'] != 'blocked-shot']
    if len(df_unblocked) > 0:
        y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        p_acc = np.clip(model.predict_proba_layer(df_unblocked, 'accuracy'), 0, 1)
        results['accuracy'] = {
            'auc': roc_auc_score(y_acc, p_acc),
            'logloss': log_loss(y_acc, p_acc),
            'brier': brier_score_loss(y_acc, p_acc)
        }
        
    # 3. Finish Layer (Target: goal | on_net)
    df_on_net = df_test[df_test['event'].isin(['shot-on-goal', 'goal'])]
    if len(df_on_net) > 0:
        y_fin = (df_on_net['event'] == 'goal').astype(int)
        p_fin = np.clip(model.predict_proba_layer(df_on_net, 'finish'), 0, 1)
        results['finish'] = {
            'auc': roc_auc_score(y_fin, p_fin),
            'logloss': log_loss(y_fin, p_fin),
            'brier': brier_score_loss(y_fin, p_fin)
        }
        
    return results

def main():
    modern_seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    
    print("Loading modern era data for benchmarking...")
    dfs = []
    for s in modern_seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            dfs.append(pd.read_csv(csv_path))
        except: pass
    df_raw = pd.concat(dfs, ignore_index=True)
    
    print("Preprocessing...")
    df = data_pipeline.preprocess_features(df_raw, is_training=True, apply_html_enrichment=False)
    
    print(f"Splitting 70/30 (Total rows: {len(df)})...")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    
    y_test_goal = (df_test['event'] == 'goal').astype(int)
    
    # Models to evaluate
    feature_list = feature_util.get_features('all_inclusive')
    architectures = [
        ("Nested GLM", fit_glm_nested.NestedGLM(features=list(feature_list), use_splines=True)),
        ("Non-Nested GLM", fit_glm.NonNestedGLM(features=list(feature_list), use_splines=True)),
        ("XGBoost Nested", fit_xgboost_nested.XGBNestedXGClassifier(features=list(feature_list), n_estimators=200)),
        ("XGBoost Non-Nested", fit_xgboost_non_nested.XGBNonNestedXGClassifier(features=list(feature_list), n_estimators=200)),
        ("XGBoost Alternate", fit_xgboost_alternate.XGBAlternateXGClassifier(features=list(feature_list), n_estimators=200))
    ]
    
    summary_rows = []
    submodel_rows = []
    
    for name, model in architectures:
        print(f"\nEvaluating {name}...")
        start_t = time.time()
        model.fit(df_train)
        fit_t = time.time() - start_t
        
        # Overall xG Performance
        probs = np.clip(model.predict_proba(df_test)[:, 1], 0, 1)
        auc = roc_auc_score(y_test_goal, probs)
        ll = log_loss(y_test_goal, probs)
        brier = brier_score_loss(y_test_goal, probs)
        calib = probs.sum() / y_test_goal.sum()
        
        summary_rows.append({
            'Model': name,
            'AUC': auc,
            'LogLoss': ll,
            'Brier': brier,
            'Calib Ratio': calib,
            'Fit Time (s)': fit_t
        })
        
        # Submodel Performance
        if hasattr(model, 'predict_proba_layer'):
            sub_res = evaluate_submodels(model, df_test)
            for layer, metrics in sub_res.items():
                submodel_rows.append({
                    'Model': name,
                    'Layer': layer,
                    'AUC': metrics['auc'],
                    'LogLoss': metrics['logloss'],
                    'Brier': metrics['brier']
                })

    # Print Summary Tables
    df_summary = pd.DataFrame(summary_rows)
    df_sub = pd.DataFrame(submodel_rows)
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE (70/30 SPLIT)")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUBMODEL PERFORMANCE")
    print("="*80)
    print(df_sub.to_string(index=False))
    
    # Save to Markdown for report
    with open('analysis/model_split_comparison.md', 'w') as f:
        f.write("# Model Comparison (70/30 Split)\n\n")
        f.write("## Overall xG Performance\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n## Submodel Layer Performance\n\n")
        f.write(df_sub.to_markdown(index=False))

if __name__ == "__main__":
    main()
