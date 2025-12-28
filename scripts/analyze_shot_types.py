
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def analyze_accuracy_layer():
    print("Loading data for Empirical Accuracy...")
    df = fit_xgs.load_all_seasons_data()
    
    # Filter for Unblocked Shots (The domain of the Accuracy Model)
    # Valid events: missed-shot, shot-on-goal, goal
    # (blocked-shot is excluded from this layer's training)
    df_unblocked = df[df['event'].isin(['missed-shot', 'shot-on-goal', 'goal'])].copy()
    df_unblocked = df_unblocked[df_unblocked['game_state'] == '5v5']
    
    df_unblocked['is_on_net'] = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)

    # 1. Empirical Accuracy Stats (Among Unblocked Attempts)
    print("\n--- Empirical Accuracy Stats (Unblocked, 5v5) ---")
    stats = df_unblocked.groupby('shot_type').agg(
        Unblocked_Attempts=('is_on_net', 'count'),
        On_Net=('is_on_net', 'sum'),
        Mean_Dist=('distance', 'mean')
    )
    stats['Accuracy_Pct'] = stats['On_Net'] / stats['Unblocked_Attempts']
    stats = stats.sort_values('Unblocked_Attempts', ascending=False)
    
    # 2. Model Controlled Test for Accuracy Layer
    print("\n--- Model Controlled Test (Accuracy Layer Only) ---")
    
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        for f in os.listdir('analysis/xgs'):
            if f.endswith('.joblib'):
                model_path = os.path.join('analysis/xgs', f)
                break

    print(f"Loading model: {model_path}")
    try:
        clf_data = joblib.load(model_path)
        # Handle if it's a tuple or object
        if isinstance(clf_data, tuple):
            model = clf_data[0]
        else:
            model = clf_data
            
        print(f"Model loaded. Accessing 'model_accuracy' layer...")
        if not hasattr(model, 'model_accuracy') or model.model_accuracy is None:
            print("Error: Model does not appear to be a fitted NestedXGClassifier (missing model_accuracy)")
            return

        # Prepare Synthetic Data (Slot Shot)
        test_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected']
        rows = []
        for st in test_types:
            r = {
                'distance': 20.0,
                'angle_deg': 0.0,
                'game_state': '5v5',
                'shot_type': st,
                'period_number': 1,
                'time_elapsed_in_period_s': 600,
                'total_time_elapsed_s': 600,
                'score_diff': 0,
                'shoots_catches': 'L',
                'event': 'shot-on-goal' 
            }
            rows.append(r)
        
        df_test = pd.DataFrame(rows)
        
        # Manually apply OHE to match what fit() did
        # We need to know the columns. The model object has 'final_features'
        # and 'config_accuracy.feature_cols'
        
        # 1. Identify Categorical Cols (from model state)
        cat_cols = model.categorical_cols
        
        # 2. Apply OHE
        df_encoded = pd.get_dummies(df_test, columns=cat_cols, prefix_sep='_')
        
        # 3. Align with model features (add missing cols as 0)
        needed_cols = model.config_accuracy.feature_cols
        for c in needed_cols:
            if c not in df_encoded.columns:
                df_encoded[c] = 0
                
        # 4. Predict using the SUB-MODEL directly
        X_test = df_encoded[needed_cols]
        probs = model.model_accuracy.predict_proba(X_test)[:, 1]
        
        results = pd.DataFrame({
            'Shot_Type': test_types,
            'Predicted_Accuracy': probs
        }).sort_values('Predicted_Accuracy', ascending=False)
        
        output_str = ""
        output_str += "--- Empirical Accuracy Stats (Unblocked, 5v5) ---\n"
        output_str += stats[['Unblocked_Attempts', 'On_Net', 'Accuracy_Pct', 'Mean_Dist']].to_string() + "\n\n"
        output_str += "--- Controlled Accuracy Prediction (P(On Net | Unblocked), 20ft Slot) ---\n"
        output_str += results.to_string()
        
        print(output_str)
        
        with open('accuracy_analysis_report.txt', 'w') as f:
            f.write(output_str)

    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_accuracy_layer()
