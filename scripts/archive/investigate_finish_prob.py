"""investigate_finish_prob.py

Script to analyze why blocked shots have high finish probabilities.
Inspects 'analysis/nested_xgs/test_predictions.csv'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def investigate():
    print("--- Investigating Finish Probability ---")
    
    csv_path = Path('analysis/nested_xgs/test_predictions.csv')
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    # Filter for Blocked Shots
    # Check if 'event' column exists, or rely on 'is_blocked'
    if 'event' in df.columns:
        blocked = df[df['event'] == 'blocked-shot'].copy()
        sog = df[df['event'] == 'shot-on-goal'].copy()
    else:
        # Fallback
        blocked = df[df['is_blocked'] == 1].copy()
        sog = df[df['is_blocked'] == 0].copy() # Approx

    print(f"Blocked Shots: {len(blocked)}")
    
    if len(blocked) == 0:
        print("No blocked shots found in predictions.")
        return

    # Sort by probability of finish
    # Columns expected: 'prob_finish', 'prob_accuracy', 'xG'
    if 'prob_finish' not in blocked.columns:
        print("Column 'prob_finish' missing.")
        return

    top_blocked = blocked.sort_values('prob_finish', ascending=False).head(20)
    
    print("\n--- Top 20 Blocked Shots by Finish Probability ---")
    cols = ['prob_finish', 'xG', 'distance', 'angle_deg', 'x', 'y', 'imputed_x', 'imputed_y', 'shot_type']
    # Check which cols exist
    show_cols = [c for c in cols if c in top_blocked.columns]
    
    print(top_blocked[show_cols])
    
    # Statistics
    print("\n--- Statistics (Finish Prob) ---")
    print(f"Blocked Mean: {blocked['prob_finish'].mean():.4f}")
    print(f"SOG Mean:     {sog['prob_finish'].mean():.4f}")
    print(f"Blocked Max:  {blocked['prob_finish'].max():.4f}")
    print(f"SOG Max:      {sog['prob_finish'].max():.4f}")
    
    # Correlation with Distance?
    print("\n--- Correlation (Blocked) ---")
    print(blocked[['prob_finish', 'distance', 'angle_deg']].corr())

    # Check Shot Type
    # Is shot_type NaN for these?
    n_nan = top_blocked['shot_type'].isna().sum()
    print(f"\nTop 20 NaN shot_type count: {n_nan}")
    if n_nan < 20:
         print("Values:", top_blocked['shot_type'].unique())

    # Visualization: Finish Prob vs Distance (Blocked vs SOG)
    plt.figure(figsize=(10, 6))
    plt.scatter(sog['distance'], sog['prob_finish'], alpha=0.1, label='SOG', s=1, color='blue')
    plt.scatter(blocked['distance'], blocked['prob_finish'], alpha=0.5, label='Blocked', s=10, color='red')
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Finish Probability')
    plt.title('Finish Probability vs Distance')
    plt.savefig('analysis/nested_xgs/finish_prob_dist.png')
    print("Saved plot to analysis/nested_xgs/finish_prob_dist.png")

if __name__ == "__main__":
    investigate()
