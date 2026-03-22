import sys
import os
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze, data_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def audit_season_features(season):
    logger.info(f"Auditing features for {season}...")
    csv_path = analyze.locate_season_csv(season)
    df = pd.read_csv(csv_path)
    
    # Use data_pipeline to get the standard features
    # is_training=False so we don't drop events arbitrarily if not needed, 
    # but we want the Standard preprocessing.
    df = data_pipeline.preprocess_features(df, is_training=False)
    
    # Filter to shot attempts
    mask_shots = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])
    df_shots = df[mask_shots].copy()
    
    # Continuous Features
    cont_cols = ['distance', 'angle_deg', 'speed']
    stats = {}
    
    for col in cont_cols:
        if col in df_shots.columns:
            series = df_shots[col].dropna()
            stats[f'{col}_mean'] = series.mean()
            stats[f'{col}_std'] = series.std()
            stats[f'{col}_median'] = series.median()
        else:
            stats[f'{col}_mean'] = np.nan
            
    # Categorical: Shot Type
    if 'shot_type' in df_shots.columns:
        counts = df_shots['shot_type'].value_counts(normalize=True)
        for s_type, pct in counts.items():
            stats[f'shot_type_{s_type}_pct'] = pct
            
    # Sample data for plotting
    sample = df_shots[['distance', 'angle_deg', 'event']].copy()
    sample['season'] = season
    
    return stats, sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', type=str, default='20202021,20212022,20222023,20232024,20242025,20252026')
    args = parser.parse_args()
    
    seasons = args.seasons.split(',')
    
    all_stats = []
    all_samples = []
    
    for s in seasons:
        try:
            stats, sample = audit_season_features(s)
            stats['season'] = s
            all_stats.append(stats)
            all_samples.append(sample.sample(min(10000, len(sample)), random_state=42))
        except Exception as e:
            logger.error(f"Error auditing {s}: {e}")
            
    stats_df = pd.DataFrame(all_stats)
    samples_df = pd.concat(all_samples)
    
    # Save
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    stats_df.to_csv(out_dir / "feature_drift_stats.csv", index=False)
    logger.info(f"Saved feature stats to {out_dir / 'feature_drift_stats.csv'}")
    
    # 1. Ridgeline Plot for Distance
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Initialize the FacetGrid object
    g = sns.FacetGrid(samples_df, row="season", hue="season", aspect=15, height=.75, palette="viridis")

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "distance", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "distance", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "distance")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    plt.savefig(out_dir / "feature_drift_dist.png")
    logger.info(f"Saved distance drift plot to {out_dir / 'feature_drift_dist.png'}")
    
    # Print Summary
    print(stats_df[['season', 'distance_mean', 'distance_std', 'angle_deg_mean']])

if __name__ == "__main__":
    main()
