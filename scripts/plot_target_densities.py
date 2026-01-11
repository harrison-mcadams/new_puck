
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def plot_dist():
    print("Loading data...")
    df = fit_xgs.load_data()
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    
    # Plot Histograms
    plt.figure(figsize=(12, 6))
    
    mask_blk = (df_shots['event'] == 'blocked-shot')
    
    plt.hist(df_shots.loc[mask_blk, 'distance'], bins=50, alpha=0.5, label='Blocked (Imputed)', color='red', density=True)
    plt.hist(df_shots.loc[~mask_blk, 'distance'], bins=50, alpha=0.5, label='Unblocked', color='blue', density=True)
    
    plt.title('Distance Distribution: Blocked vs Unblocked')
    plt.xlabel('Distance (ft)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('analysis/diagnostic_distance_density.png')
    print("Saved plot to analysis/diagnostic_distance_density.png")

if __name__ == "__main__":
    plot_dist()
