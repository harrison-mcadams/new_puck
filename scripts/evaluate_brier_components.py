import sys
import os
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def decompose_brier(y_true, y_pred, n_bins=10):
    """
    Decomposes Brier Score into Reliability, Resolution, and Uncertainty.
    Brier = Rel - Res + Unc
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = len(y_true)
    
    # 1. Uncertainty: global_mean * (1 - global_mean)
    global_mean = np.mean(y_true)
    uncertainty = global_mean * (1 - global_mean)
    
    # 2. Binning
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    
    reliability = 0.0
    resolution = 0.0
    
    bin_centers = []
    bin_actuals = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (binids == i)
        if not np.any(mask):
            continue
            
        count_i = np.sum(mask)
        mean_pred_i = np.mean(y_pred[mask])
        mean_actual_i = np.mean(y_true[mask])
        
        reliability += count_i * (mean_pred_i - mean_actual_i)**2
        resolution += count_i * (mean_actual_i - global_mean)**2
        
        bin_centers.append(mean_pred_i)
        bin_actuals.append(mean_actual_i)
        bin_counts.append(count_i)
        
    reliability /= total
    resolution /= total
    
    brier = np.mean((y_true - y_pred)**2)
    
    return {
        'brier': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'sum_check': reliability - resolution + uncertainty,
        'bin_centers': bin_centers,
        'bin_actuals': bin_actuals,
        'bin_counts': bin_counts
    }

def run_calibration_audit(seasons):
    evaluator = PredictiveEvaluator(
        model_name='nested_xg',
        metric_type='gd',
        filter_type='all',
        n_boot=0 # No bootstrapping needed for raw decomposition
    )
    
    results = []
    calibration_data = {}
    
    for season in seasons:
        logger.info(f"Auditing Brier components for {season}...")
        res = evaluator.run_evaluation(season, train_split=0.7, split_method='random', n_reps=1)
        if res is None: continue
        
        raw = res['Raw_Results']
        decomp = decompose_brier(raw['y'], raw['p'])
        
        entry = {
            'season': season,
            'brier': decomp['brier'],
            'reliability': decomp['reliability'],
            'resolution': decomp['resolution'],
            'uncertainty': decomp['uncertainty']
        }
        results.append(entry)
        calibration_data[season] = decomp
        
    res_df = pd.DataFrame(results)
    return res_df, calibration_data

def plot_calibration(calibration_data, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unity line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
    
    # Use a colormap for seasons
    seasons = sorted(calibration_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(seasons)))
    
    for i, season in enumerate(seasons):
        data = calibration_data[season]
        ax.plot(data['bin_centers'], data['bin_actuals'], 'o-', label=season, color=colors[i])
        
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Mean Actual Outcome')
    ax.set_title('Calibration Curves by Season (Nested xG)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved calibration panel to {out_path}")

def plot_brier_decomposition(res_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by season
    res_df = res_df.sort_values('season')
    
    x = np.arange(len(res_df))
    
    # We'll plot them as lines or stacked bars.
    # Lines show trends better.
    ax.plot(res_df['season'], res_df['reliability'], marker='o', label='Reliability (Lower is better)')
    ax.plot(res_df['season'], res_df['resolution'], marker='s', label='Resolution (Higher is better)')
    ax.plot(res_df['season'], res_df['uncertainty'], marker='^', label='Uncertainty (Parity proxy)')
    ax.plot(res_df['season'], res_df['brier'], marker='D', color='black', linewidth=2, label='Brier Total')
    
    ax.set_ylabel('Score Component')
    ax.set_title('Brier Score Decomposition over Time')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Brier decomposition plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', type=str, default='20202021,20212022,20222023,20232024,20242025,20252026')
    args = parser.parse_args()
    
    seasons = args.seasons.split(',')
    
    res_df, cal_data = run_calibration_audit(seasons)
    
    # Save metrics
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    res_df.to_csv(out_dir / "brier_decomposition.csv", index=False)
    
    # Plots
    plot_calibration(cal_data, out_dir / "calibration_panel.png")
    plot_brier_decomposition(res_df, out_dir / "brier_decomposition_trend.png")
    
    print(res_df)

if __name__ == "__main__":
    main()
