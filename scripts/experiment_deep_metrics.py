
import sys
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck import fit_nested_xgs, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DeepMetrics")

def analyze_layer_performance(y_true, y_pred, name):
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    logger.info(f"[{name}] AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}")
    
    # Calibration Bias
    mean_pred = np.mean(y_pred)
    mean_true = np.mean(y_true)
    logger.info(f"[{name}] Mean Pred: {mean_pred:.4f} | Mean True: {mean_true:.4f} | Diff: {mean_true - mean_pred:.4f}")
    return auc, ll

def main():
    logger.info("Loading Data...")
    df = fit_nested_xgs.load_data()
    # Sample down for speed for this final summary
    df = df.sample(frac=0.1, random_state=42)
    
    df = fit_nested_xgs.preprocess_features(df)
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # Split
    X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)
    
    # Configs to test
    configs = [
        ("Baseline", None, None, None),
        ("Balanced All", 'balanced_subsample', 'balanced_subsample', 'balanced_subsample'),
        ("Balanced Finish Only", None, None, 'balanced_subsample')
    ]
    
    results = []
    
    for name, bp_w, ap_w, fp_w in configs:
        logger.info(f"\n--- Testing Config: {name} ---")
        
        bp = {'class_weight': bp_w} if bp_w else {}
        ap = {'class_weight': ap_w} if ap_w else {}
        fp = {'class_weight': fp_w} if fp_w else {} # Finish is always the critical one
        
        clf = fit_nested_xgs.NestedXGClassifier(
            n_estimators=100, max_depth=10,
            block_params=bp, accuracy_params=ap, finish_params=fp
        )
        clf.fit(X_train)
        
        # 1. Block Layer Performance (internal model access)
        # Note: We need to recreate the block target
        X_test_block = X_test.copy()
        y_test_block = X_test_block['is_blocked']
        
        probs_block = clf.predict_proba_layer(X_test_block, 'block')
        
        logger.info(f"Block Layer Performance ({name}):")
        auc_b, ll_b = analyze_layer_performance(y_test_block, probs_block, f"Block-{name}")
        
        # 2. Global xG Performance
        probs_xg = clf.predict_proba(X_test)[:, 1]
        y_test_goal = (X_test['event'] == 'goal').astype(int)
        
        logger.info(f"Global xG Performance ({name}):")
        auc_g, ll_g = analyze_layer_performance(y_test_goal, probs_xg, f"Global-{name}")
        
        results.append({
            'Model': name,
            'Block_AUC': auc_b,
            'Block_Bias': np.mean(probs_block) - np.mean(y_test_block), # Pos = Overpredict Block
            'Global_AUC': auc_g,
            'Global_LogLoss': ll_g
        })

    logger.info("\n=== SUMMARY RESULTS ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string())
    
    # Recommendation calculation
    best_auc = res_df.sort_values('Global_AUC', ascending=False).iloc[0]
    logger.info(f"\nBest AUC Model: {best_auc['Model']}")
    
    # Check if Balanced All broke Blocks
    bal_all = res_df[res_df['Model'] == 'Balanced All'].iloc[0]
    if abs(bal_all['Block_Bias']) > 0.05:
        logger.warning(f"WARNING: 'Balanced All' has significant Block Bias ({bal_all['Block_Bias']:.4f}). Consider 'Balanced Finish Only'.")

if __name__ == "__main__":
    main()
