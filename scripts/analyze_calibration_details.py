
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

def analyze_calibration():
    # Load predictions
    # try enriched first, then bare
    try:
        df = pd.read_csv('analysis/nested_xgs/test_predictions.csv')
        print("Loaded enriched predictions.")
    except:
        df = pd.read_csv('analysis/nested_xgs/test_predictions_bare.csv')
        print("Loaded bare predictions.")
        
    print(f"Rows: {len(df)}")
    
    # Define Targets
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    
    # --- Helper to print calibration table ---
    def print_calib_table(y_true, y_prob, name):
        print(f"\n--- {name} Calibration ---")
        print(f"Overall Avg Prob:  {y_prob.mean():.4f}")
        print(f"Overall Actual:    {y_true.mean():.4f}")
        print(f"Ratio (Pred/Act):  {y_prob.mean() / y_true.mean():.4f}")
        print(f"Log Loss:          {log_loss(y_true, y_prob):.4f}")
        
        # Bins
        bins = np.linspace(0, 1, 11)
        df_temp = pd.DataFrame({'prob': y_prob, 'target': y_true})
        df_temp['bin'] = pd.cut(df_temp['prob'], bins)
        
        grouped = df_temp.groupby('bin', observed=False).agg(
            mean_prob=('prob', 'mean'),
            mean_target=('target', 'mean'),
            count=('target', 'count')
        )
        print(grouped)

    # 1. Overall xG
    print_calib_table(df['is_goal'], df['xG'], "Overall xG")
    
    # 2. Block Model
    # Predicted 'prob_block' vs 'is_blocked'
    if 'prob_block' in df.columns:
        print_calib_table(df['is_blocked'], df['prob_block'], "Block Model")
        
    # 3. Accuracy Model (Subset: Unblocked)
    mask_unblocked = df['is_blocked'] == 0
    if mask_unblocked.any() and 'prob_accuracy' in df.columns:
        sub = df[mask_unblocked]
        print_calib_table(sub['is_on_net'], sub['prob_accuracy'], "Accuracy Model (Unblocked Only)")
        
    # 4. Finish Model (Subset: On Net)
    mask_on_net = (df['is_blocked'] == 0) & (df['event'].isin(['shot-on-goal', 'goal']))
    if mask_on_net.any() and 'prob_finish' in df.columns:
        sub = df[mask_on_net]
        print_calib_table(sub['is_goal'], sub['prob_finish'], "Finish Model (On Net Only)")

if __name__ == "__main__":
    analyze_calibration()
