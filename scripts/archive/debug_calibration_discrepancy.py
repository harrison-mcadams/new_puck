
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, analyze, config, fit_nested_xgs

def main():
    print("Layer-by-Layer Calibration Analysis...")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        model_path = 'analysis/xgs/xg_model_nested.joblib'
    clf = joblib.load(model_path)
    
    # 2. Load Data
    print("Loading all data...")
    df_raw = fit_xgs.load_all_seasons_data()
    
    # Preprocess (Standard Regular Season Exclusions)
    df = fit_nested_xgs.preprocess_features(df_raw)
    from puck import impute
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # Add season
    df['season'] = df['game_id'].astype(str).str[:4]
    
    # 3. Predict Layers
    print("Predicting Layer Probabilities...")
    p_block = clf.predict_proba_layer(df, 'block')
    p_acc = clf.predict_proba_layer(df, 'accuracy')
    p_fin = clf.predict_proba_layer(df, 'finish')
    
    df['p_unblocked'] = 1.0 - p_block
    df['p_on_net'] = p_acc
    df['p_goal_given_on_net'] = p_fin
    df['p_goal'] = df['p_unblocked'] * df['p_on_net'] * df['p_goal_given_on_net']
    
    # 4. Calibration by Layer and Season (5v5 Only)
    df_5v5 = df[df['game_state'] == '5v5'].copy()
    
    def calc_layer_metrics(pdf):
        # Block Layer
        actual_unblocked = (pdf['event'] != 'blocked-shot').mean()
        pred_unblocked = pdf['p_unblocked'].mean()
        
        # Accuracy Layer (Condition: Unblocked)
        df_ub = pdf[pdf['event'] != 'blocked-shot']
        actual_on_net = df_ub['event'].isin(['shot-on-goal', 'goal']).mean()
        pred_on_net = df_ub['p_on_net'].mean()
        
        # Finish Layer (Condition: On Net)
        df_on = df_ub[df_ub['event'].isin(['shot-on-goal', 'goal'])]
        actual_goal = (df_on['event'] == 'goal').mean()
        pred_goal = df_on['p_goal_given_on_net'].mean()
        
        total_xg = pdf['p_goal'].sum()
        total_goals = (pdf['event'] == 'goal').sum()
        
        return pd.Series({
            'Unblocked_Act': actual_unblocked,
            'Unblocked_Pred': pred_unblocked,
            'Unblocked_Ratio': pred_unblocked / actual_unblocked,
            'OnNet_Act': actual_on_net,
            'OnNet_Pred': pred_on_net,
            'OnNet_Ratio': pred_on_net / actual_on_net,
            'Goal_Act': actual_goal,
            'Goal_Pred': pred_goal,
            'Goal_Ratio': pred_goal / actual_goal,
            'Overall_Ratio': total_xg / total_goals if total_goals > 0 else 0
        })

    summary = df_5v5.groupby('season').apply(calc_layer_metrics)
    
    print("\n5v5 Layer Calibration by Season:")
    print(summary.round(4).to_markdown())

if __name__ == "__main__":
    main()
