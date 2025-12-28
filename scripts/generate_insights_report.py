
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def generate_insights():
    print("Loading all seasons data...")
    df = fit_xgs.load_all_seasons_data()
    
    # ---------------------------------------------------------
    # 1. SETUP & FILTERING
    # ---------------------------------------------------------
    # We focus on 5v5 for "fair" empirical stats
    df_5v5 = df[df['game_state'] == '5v5'].copy()
    
    # Define event buckets
    # Block Layer: All attempts (shot, goal, miss, block)
    attempts = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_attempts = df_5v5[df_5v5['event'].isin(attempts)].copy()
    
    # Accuracy Layer: Unblocked attempts (shot, goal, miss)
    unblocked_events = ['shot-on-goal', 'goal', 'missed-shot']
    df_unblocked = df_5v5[df_5v5['event'].isin(unblocked_events)].copy()
    
    # Finish Layer: On-Net attempts (shot, goal)
    on_net_events = ['shot-on-goal', 'goal']
    df_on_net = df_5v5[df_5v5['event'].isin(on_net_events)].copy()
    
    # Add targets
    df_attempts['is_blocked'] = (df_attempts['event'] == 'blocked-shot').astype(int)
    df_unblocked['is_on_net'] = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
    df_on_net['is_goal'] = (df_on_net['event'] == 'goal').astype(int)

    # ---------------------------------------------------------
    # 2. LOAD MODEL
    # ---------------------------------------------------------
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        # Fallback search
        found = False
        for f in os.listdir('analysis/xgs'):
            if f.endswith('.joblib'):
                model_path = os.path.join('analysis/xgs', f)
                found = True
                break
        if not found:
            print("No model found.")
            return

    print(f"Loading model: {model_path}")
    clf_data = joblib.load(model_path)
    model = clf_data[0] if isinstance(clf_data, tuple) else clf_data

    if not hasattr(model, 'model_block'):
        print("Model is not a NestedXGClassifier.")
        return

    # ---------------------------------------------------------
    # 3. HELPER: PREDICT FOR CONTROLLED SCENARIOS
    # ---------------------------------------------------------
    def get_controlled_preds(dist_ft, angle_deg, layer_name):
        """Predict prob for all shot types at fixed location."""
        shot_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected']
        rows = []
        for st in shot_types:
            r = {
                'distance': float(dist_ft),
                'angle_deg': float(angle_deg),
                'game_state': '5v5',
                'shot_type': st,
                'period_number': 1,
                'time_elapsed_in_period_s': 600,
                'total_time_elapsed_s': 600,
                'score_diff': 0,
                'shoots_catches': 'L',
                # Dummy event for feature extraction
                'event': 'shot-on-goal'
            }
            rows.append(r)
        
        df_test = pd.DataFrame(rows)
        
        # OHE Logic (Mimic NestedXGClassifier.fit/predict internals)
        cat_cols = model.categorical_cols
        if cat_cols:
            df_encoded = pd.get_dummies(df_test, columns=cat_cols, prefix_sep='_')
        else:
            df_encoded = df_test.copy()
            
        sub_model = None
        needed_cols = []
        
        if layer_name == 'block':
            sub_model = model.model_block
            needed_cols = model.config_block.feature_cols
        elif layer_name == 'accuracy':
            sub_model = model.model_accuracy
            needed_cols = model.config_accuracy.feature_cols
        elif layer_name == 'finish':
            sub_model = model.model_finish
            needed_cols = model.config_finish.feature_cols
        elif layer_name == 'total':
             # Use the main predict_proba wrapper
             pass
             
        # Align Columns
        if layer_name != 'total':
            for c in needed_cols:
                if c not in df_encoded.columns:
                    df_encoded[c] = 0
            X_test = df_encoded[needed_cols]
            return sub_model.predict_proba(X_test)[:, 1]
        else:
            # Main model prediction
            return model.predict_proba(df_test)[:, 1]

    # ---------------------------------------------------------
    # 4. GENERATE REPORT SECTIONS
    # ---------------------------------------------------------
    
    report = []
    report.append("# Nested xG Model: Comprehensive Performance Insights\n")
    report.append("**Model:** Nested_All (2014-2026 Data)\n")
    report.append("**Generated:** Automatically from `scripts/generate_insights_report.py`\n")
    report.append("---\n")

    # --- LAYER 1: BLOCK MODEL ---
    report.append("## Layer 1: Block Model")
    report.append("*Predicts: Probability of a shot being blocked (given it was attempted).*")
    
    # Empirical
    stats_block = df_attempts.groupby('shot_type').agg(
        Attempts=('is_blocked', 'count'),
        Blocked=('is_blocked', 'sum'),
        Mean_Dist=('distance', 'mean')
    )
    stats_block['Block_Pct'] = stats_block['Blocked'] / stats_block['Attempts']
    stats_block = stats_block.sort_values('Attempts', ascending=False)
    
    report.append("\n### Empirical Data (5v5)")
    report.append(stats_block[['Attempts', 'Block_Pct', 'Mean_Dist']].round(3).to_markdown())
    
    # Controlled
    preds_block = get_controlled_preds(20, 0, 'block') # Slot
    res_block = pd.DataFrame({'Shot_Type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected'], 'Pred_Block_Prob_Slot': preds_block})
    preds_block_point = get_controlled_preds(50, 0, 'block') # Point
    res_block['Pred_Block_Prob_Point'] = preds_block_point
    
    report.append("\n### Model Logic (Controlled Test)")
    report.append("Comparing a Slot Shot (20ft) vs Point Shot (50ft):")
    report.append(res_block.set_index('Shot_Type').round(3).to_markdown())
    
    report.append("\n> **Insight:** Note how different shot types have different 'blockability' even at the same distance.")
    report.append("---\n")

    # --- LAYER 2: ACCURACY MODEL ---
    report.append("## Layer 2: Accuracy Model")
    report.append("*Predicts: Probability of hitting the net (given it was NOT blocked).*")
    
    # Empirical
    stats_acc = df_unblocked.groupby('shot_type').agg(
        Unblocked=('is_on_net', 'count'),
        On_Net=('is_on_net', 'sum'),
        Mean_Dist=('distance', 'mean')
    )
    stats_acc['Accuracy_Pct'] = stats_acc['On_Net'] / stats_acc['Unblocked']
    stats_acc = stats_acc.sort_values('Unblocked', ascending=False)
    
    report.append("\n### Empirical Data (5v5)")
    report.append(stats_acc[['Unblocked', 'Accuracy_Pct', 'Mean_Dist']].round(3).to_markdown())

    # Controlled
    preds_acc = get_controlled_preds(20, 0, 'accuracy')
    res_acc = pd.DataFrame({'Shot_Type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected'], 'Pred_Accuracy_Slot': preds_acc})
    
    report.append("\n### Model Logic (Controlled Test)")
    report.append("Predicted Accuracy from the Slot (20ft):")
    report.append(res_acc.set_index('Shot_Type').round(3).to_markdown())
    report.append("\n> **Insight:** Wrist shots are significantly more accurate than slap shots. Tip-ins are low accuracy because they are deflections.")
    report.append("---\n")

    # --- LAYER 3: FINISH MODEL ---
    report.append("## Layer 3: Finish Model")
    report.append("*Predicts: Probability of scoring (given it is On Net).*")
    
    # Empirical
    stats_fin = df_on_net.groupby('shot_type').agg(
        On_Net=('is_goal', 'count'),
        Goals=('is_goal', 'sum'),
        Mean_Dist=('distance', 'mean')
    )
    stats_fin['Shooting_Pct'] = stats_fin['Goals'] / stats_fin['On_Net']
    stats_fin = stats_fin.sort_values('On_Net', ascending=False)
    
    report.append("\n### Empirical Data (5v5)")
    report.append(stats_fin[['On_Net', 'Shooting_Pct', 'Mean_Dist']].round(3).to_markdown())

    # Controlled
    preds_fin = get_controlled_preds(20, 0, 'finish')
    res_fin = pd.DataFrame({'Shot_Type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected'], 'Pred_Shooting_Pct_Slot': preds_fin})
    
    report.append("\n### Model Logic (Controlled Test)")
    report.append("Predicted Shooting % if On Net (Slot, 20ft):")
    report.append(res_fin.set_index('Shot_Type').round(3).to_markdown())
    report.append("\n> **Insight:** Here is where Slap Shots shine. If they hit the net, they are harder to save.")
    report.append("---\n")

    # --- SUMMARY: TOTAL xG ---
    report.append("## Summary: Total xG")
    report.append("*P(Goal) = P(Unblocked) * P(On Net) * P(Score)*")
    
    preds_total = get_controlled_preds(20, 0, 'total')
    res_total = pd.DataFrame({
        'Shot_Type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected'], 
        'Total_xG_Slot': preds_total
    })
    
    report.append("\n### Combined xG (Slot, 20ft)")
    report.append(res_total.set_index('Shot_Type').round(4).to_markdown())
    
    # Save
    with open('model_insights.md', 'w') as f:
        f.write("\n".join(report))
    
    print("Report saved to model_insights.md")

if __name__ == "__main__":
    generate_insights()
