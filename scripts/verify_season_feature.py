import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import joblib

from puck import data_pipeline, features as feature_util
from puck.fit_xgboost_non_nested import XGBNonNestedXGClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_modern_era_data():
    """Load data from multiple modern era seasons."""
    season_paths = {
        '20212022': 'data/20212022/20212022_df.csv',
        '20222023': 'data/20222023/20222023_df.csv',
        '20232024': 'data/20232024.csv',
        '20242025': 'data/20242025/20242025_df.csv',
        '20252026': 'data/20252026.csv'
    }
    all_dfs = []
    for s, p in season_paths.items():
        path = Path(p)
        if path.exists():
            logger.info(f"Loading season {s} from {p}...")
            df = pd.read_csv(path)
            # Ensure season column exists and is string
            df['season'] = str(s)
            all_dfs.append(df)
        else:
            logger.warning(f"Season {s} data not found at {path}")
    
    if not all_dfs:
        raise FileNotFoundError("No modern era data found!")
        
    return pd.concat(all_dfs, ignore_index=True)

def run_study(n_iterations=100):
    df_raw = load_modern_era_data()
    
    # Preprocess once for everything
    logger.info("Preprocessing data...")
    df = data_pipeline.preprocess_features(
        df_raw, 
        is_training=True, 
        verbose=False,
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=True,
        apply_filtering=True
    )
    
    # Define feature sets
    features_with_season = feature_util.get_features('all_inclusive')
    features_no_season = [f for f in features_with_season if f != 'season']
    
    logger.info(f"Features (With Season): {features_with_season}")
    logger.info(f"Features (No Season): {features_no_season}")
    
    results = []
    
    # Per-season aggregate tracking
    season_tracking = []

    for i in range(n_iterations):
        if i % 10 == 0:
            logger.info(f"Iteration {i}/{n_iterations}...")
        
        # Shuffle and Split
        df_shuffled = df.sample(frac=1, random_state=i).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * 0.8)
        df_train = df_shuffled.iloc[:split_idx]
        df_test = df_shuffled.iloc[split_idx:]
        
        y_test = (df_test['event'] == 'goal').astype(int)
        
        # 1. Model A: No Season
        model_a = XGBNonNestedXGClassifier(features=features_no_season, n_estimators=100, max_depth=4, learning_rate=0.1)
        model_a.fit(df_train)
        probs_a = model_a.predict_proba(df_test)[:, 1]
        
        # 2. Model B: With Season
        model_b = XGBNonNestedXGClassifier(features=features_with_season, n_estimators=100, max_depth=4, learning_rate=0.1)
        model_b.fit(df_train)
        probs_b = model_b.predict_proba(df_test)[:, 1]
        
        # Metrics
        from sklearn.metrics import brier_score_loss, log_loss
        results.append({
            'iteration': i,
            'brier_no_season': brier_score_loss(y_test, probs_a),
            'brier_with_season': brier_score_loss(y_test, probs_b),
            'logloss_no_season': log_loss(y_test, probs_a),
            'logloss_with_season': log_loss(y_test, probs_b)
        })
        
        # Aggregate xG vs Actual per season in this test set
        test_agg = df_test.copy()
        test_agg['xg_no_season'] = probs_a.tolist()
        test_agg['xg_with_season'] = probs_b.tolist()
        test_agg['is_goal'] = y_test.values.tolist()
        
        agg = test_agg.groupby('season').agg({
            'is_goal': 'sum',
            'xg_no_season': 'sum',
            'xg_with_season': 'sum'
        }).reset_index()
        agg['iteration'] = i
        season_tracking.append(agg)
        
    results_df = pd.DataFrame(results)
    season_df = pd.concat(season_tracking)
    
    # Print Summary Metrics
    logger.info("\n--- Performance Summary (%d Iterations) ---" % n_iterations)
    logger.info("Brier (No Season):   %.6f +/- %.6f" % (results_df['brier_no_season'].mean(), results_df['brier_no_season'].std()))
    logger.info("Brier (With Season): %.6f +/- %.6f" % (results_df['brier_with_season'].mean(), results_df['brier_with_season'].std()))
    logger.info("LogLoss (No Season):   %.4f +/- %.4f" % (results_df['logloss_no_season'].mean(), results_df['logloss_no_season'].std()))
    logger.info("LogLoss (With Season): %.4f +/- %.4f" % (results_df['logloss_with_season'].mean(), results_df['logloss_with_season'].std()))
    
    # Generate Plot (Matplotlib only)
    seasons = sorted(season_df['season'].unique())
    metrics_to_plot = ['is_goal', 'xg_no_season', 'xg_with_season']
    labels = ['Actual Goals', 'XGBoost (Base)', 'XGBoost (+Season)']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    x = np.arange(len(seasons))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, m in enumerate(metrics_to_plot):
        data_mean = [season_df[season_df['season'] == s][m].mean() for s in seasons]
        data_std = [season_df[season_df['season'] == s][m].std() for s in seasons]
        ax.bar(x + (i - 1) * width, data_mean, width, label=labels[i], color=colors[i], yerr=data_std, capsize=5)
    
    ax.set_title(f"Expected vs Actual Goals per Season\nAggregate over {n_iterations} bootstrap iterations", fontsize=14)
    ax.set_ylabel("Total Goals / xG", fontsize=12)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = "analysis/season_feature_impact.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")
    
    results_df.to_csv("analysis/season_study_metrics.csv", index=False)
    season_df.to_csv("analysis/season_study_aggregates.csv", index=False)

if __name__ == "__main__":
    run_study(n_iterations=25)
