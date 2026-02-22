import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from pathlib import Path
import joblib

def main():
    # 1. Load Data
    data_path = Path("analysis/season_shots_20252026.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Check "Too Good To Be True" Metrics
    y_true = (df['event'] == 'goal').astype(int)
    
    # Base Model (xg_nested)
    auc_base = roc_auc_score(y_true, df['xg_nested'])
    loss_base = log_loss(y_true, df['xg_nested'])
    
    # Mixed Effects (xtg_mixed)
    auc_mixed = roc_auc_score(y_true, df['xtg_mixed'])
    loss_mixed = log_loss(y_true, df['xtg_mixed'])
    
    print(f"\n--- Performance Metrics ---")
    print(f"Base Model:   AUC={auc_base:.4f}, LogLoss={loss_base:.4f}")
    print(f"Mixed Effects: AUC={auc_mixed:.4f}, LogLoss={loss_mixed:.4f}") # If AUC > 0.9, very suspicious
    
    # 3. Inspect High Probability Events
    # User says "recognizing goals with very high probability"
    # Let's look at the distribution of xtg for GOALS
    
    goals = df[df['event'] == 'goal']
    print(f"\n--- Goal Predictions Stats (xtg_mixed) ---")
    print(goals['xtg_mixed'].describe())
    
    high_prob_goals = goals[goals['xtg_mixed'] > 0.5]
    print(f"Goals with xtg > 0.5: {len(high_prob_goals)} / {len(goals)} ({len(high_prob_goals)/len(goals):.1%})")
    
    very_high_prob = goals[goals['xtg_mixed'] > 0.8]
    print(f"Goals with xtg > 0.8: {len(very_high_prob)} / {len(goals)} ({len(very_high_prob)/len(goals):.1%})")
    
    # 4. Leakage Check?
    # Are there features that are uniquely identifying?
    # Let's load the model and check coefficients
    model_path = Path("analysis/xgs/joint_mixed_effects.joblib")
    if model_path.exists():
        print(f"\n--- Loading Model for Coefficient Inspection ---")
        try:
            # We need the class defs available. 
            # They should be imported by joblib if puck package is installed/in path
            # But let's append path just in case
            import sys
            import os
            sys.path.append(os.getcwd())
            
            model = joblib.load(model_path)
            
            # Check coefficients for 5v5 Offense
            if '5v5' in model.off_models_:
                off_5v5 = model.off_models_['5v5']
                coefs = off_5v5.get_coefficients()
                
                print("\nTop 10 Absolute Coefficients (5v5 Offense):")
                coefs['abs_coef'] = coefs['coef'].abs()
                print(coefs.sort_values('abs_coef', ascending=False).head(10))
                
                # Check for extreme values
                extreme_coefs = coefs[coefs['abs_coef'] > 5.0]
                if not extreme_coefs.empty:
                    print(f"\nWARNING: Found {len(extreme_coefs)} coefficients > 5.0")
                    print(extreme_coefs)
                else:
                    print("\nNo coefficients > 5.0 found (Good).")
                    
        except Exception as e:
            print(f"Error loading model: {e}")

    # 5. Plotting Distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='xtg_mixed', hue='event', bins=50, kde=True, log_scale=(False, True))
    plt.title("Distribution of xtg_mixed (Log Scale)")
    
    plt.subplot(1, 2, 2)
    # Scatter vs Base
    sns.scatterplot(x=df['xg_nested'], y=df['xtg_mixed'], hue=df['event'], alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Base vs Mixed (Hue=Event)")
    
    plot_path = Path("analysis/plots/overfitting_audit.png")
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved audit plot to {plot_path}")
    
    # Save metrics to text file for easy reading
    with open("audit_results.txt", "w") as f:
        f.write(f"Base AUC: {auc_base:.4f}\n")
        f.write(f"Mixed AUC: {auc_mixed:.4f}\n")
        f.write(f"Base LogLoss: {loss_base:.4f}\n")
        f.write(f"Mixed LogLoss: {loss_mixed:.4f}\n")
        f.write(f"Goals > 0.5: {len(high_prob_goals)}\n")
        f.write(f"Goals > 0.8: {len(very_high_prob)}\n")

    import sys
    sys.stdout.flush()

if __name__ == "__main__":
    main()
