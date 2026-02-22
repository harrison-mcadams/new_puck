
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys
import os

def main():
    # Load Model
    model_path = Path("analysis/xgs/joint_mixed_effects.joblib")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        sys.path.append(os.getcwd())
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract Coefficients
    print("Extracting coefficients...")
    df_coefs = model.get_all_coefficients()
    
    if df_coefs.empty:
        print("No coefficients found in model.")
        return

    # Filter for Intercepts
    df_intercepts = df_coefs[df_coefs['feature'] == 'intercept'].copy()
    
    if df_intercepts.empty:
        print("No intercept coefficients found.")
        return

    # Setup Output
    output_dir = Path("analysis/plots/intercepts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting Style
    sns.set_theme(style="whitegrid")
    
    # Process each Game State (e.g., 5v5)
    states = df_intercepts['game_state'].unique()
    
    for state in states:
        print(f"Plotting for state: {state}")
        
        # 1. Offense
        df_off = df_intercepts[(df_intercepts['game_state'] == state) & 
                               (df_intercepts['role'] == 'Offense')]
        
        if not df_off.empty:
            # Sort: Higher is Better for Offense
            df_off = df_off.sort_values('coef', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df_off, x='coef', y='team', palette='viridis')
            plt.title(f"Team Offense Intercepts ({state})\nHigher = More Goals Above Expectation")
            plt.xlabel("Log-Odds Adjustment")
            plt.ylabel("Team")
            plt.tight_layout()
            save_path = output_dir / f"intercepts_{state}_offense.png"
            plt.savefig(save_path)
            print(f"Saved {save_path}")
            plt.close()

        # 2. Defense
        df_def = df_intercepts[(df_intercepts['game_state'] == state) & 
                               (df_intercepts['role'] == 'Defense')]
        
        if not df_def.empty:
            # Sort: Lower is Better for Defense (prevents goals)
            # But for plotting "Performance", maybe we want "Better" at top?
            # Let's stick to coefficient value for truth, but arguably 
            # sorting ascending (most negative first) puts best defensive teams at top?
            # Yes, let's put most negative (best defense) at top.
            df_def = df_def.sort_values('coef', ascending=True)
            
            plt.figure(figsize=(12, 8))
            # Use a different palette for defense (e.g., coolwarm_r)
            sns.barplot(data=df_def, x='coef', y='team', palette='coolwarm')
            plt.title(f"Team Defense Intercepts ({state})\nLower (Negative) = Fewer Goals Against Expectation")
            plt.xlabel("Log-Odds Adjustment")
            plt.ylabel("Team")
            plt.tight_layout()
            save_path = output_dir / f"intercepts_{state}_defense.png"
            plt.savefig(save_path)
            print(f"Saved {save_path}")
            plt.close()

if __name__ == "__main__":
    main()
