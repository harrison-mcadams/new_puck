
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import impute

def verify_alpha_effect():
    print("Verifying Alpha Knob Effect...")
    
    # Create a single dummy blocked shot in the High Slot (65, 0)
    # This is our "Aggressive" point.
    df_block = pd.DataFrame({
        'event': ['blocked-shot'] * 1000,
        'x': [65.0] * 1000,
        'y': [0.0] * 1000,
        'shooter_role': ['F'] * 1000,
        'distance': [24.0] * 1000 # Distance to net (89 - 65)
    })
    
    alphas = [0.0, 0.2, 0.5, 0.8]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
    for alpha, ax in zip(alphas, axes):
        # Run imputation
        df_res = impute.impute_blocked_shot_origins(
            df_block.copy(), 
            method='mixture_model', 
            alpha=alpha,
            is_standardized=True
        )
        
        # Plot
        ax.scatter(df_res['imputed_x'], df_res['imputed_y'], alpha=0.1, s=5, color='blue')
        ax.scatter(65, 0, color='red', marker='x', label='Block Loc') # Original Block
        ax.set_title(f"Alpha = {alpha}")
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = os.path.join('analysis', 'mixture_model_verification.png')
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(out_path)
    print(f"  Verification plot saved to {out_path}")

if __name__ == "__main__":
    verify_alpha_effect()
