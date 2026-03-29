"""modern_era_benchmarking_suite.py

Comprehensive benchmarking for NHL xG models across the Modern Era (2020-2026).
Compares:
1. Nested XGBoost (nested_xg)
2. Non-Nested XGBoost (non_nested_xg)
3. Nested GLM (nested)
4. Non-Nested GLM (non_nested)
5. Actual Goals baseline (actual)

Analyses:
- Predictive Power (Brier Score, Accuracy)
- Stability (Hockey-Graphs Reliability R²)
- Forecasting (Season-level R² for Wins and GD)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import evaluation components
from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = ['nested_xg', 'non_nested_xg', 'nested', 'non_nested', 'actual']
FILTERS = ['all', '5v5']

# Professional Color Palette
PALETTE = {
    'nested_xg': '#0055CC',      # Deep Blue
    'non_nested_xg': '#CC5500',  # Burnt Orange
    'nested': '#0099CC',         # Cyan
    'non_nested': '#CCAA00',     # Amber
    'actual': '#000000',         # Black (Baseline)
}

def plot_benchmark_results(all_metrics, stability_metrics, prediction_metrics, out_dir, mode='violin'):
    """Generates high-fidelity comparison plots (Violin, Box, or Bar)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    
    # helper to create distribution dataframe
    def build_dist_df(metrics_list, dist_key, val_name):
        rows = []
        for m in metrics_list:
            dist = m.get(dist_key, [])
            if len(dist) == 0:
                # Fallback to mean if no distribution exists
                rows.append({'Model': m['Model'], 'Filter': m['Filter'], val_name: m.get(val_name, 0.0)})
            else:
                for val in dist:
                    rows.append({'Model': m['Model'], 'Filter': m['Filter'], val_name: val})
        return pd.DataFrame(rows)

    # --- 1. Predictive Power Plots ---
    if all_metrics:
        brier_df = build_dist_df(all_metrics, 'Brier_dist', 'Brier')
        acc_df = build_dist_df(all_metrics, 'Acc_dist', 'Acc')
        
        for name, df, ylabel, title, out_name, ylim in [
            ('Brier', brier_df, 'Brier Score (Lower is Better)', 'Predictive Power: Brier Score Distribution', 'modern_benchmark_brier.png', (0.22, 0.255)),
            ('Acc', acc_df, 'Accuracy (%)', 'Predictive Power: Game Accuracy Distribution', 'modern_benchmark_accuracy.png', (0.50, 0.65))
        ]:
            if df.empty: continue
            plt.figure(figsize=(11, 6))
            if mode == 'violin':
                sns.violinplot(data=df, x='Filter', y=name, hue='Model', palette=PALETTE, inner='quartile', split=False)
                # Overlay means as points
                sns.pointplot(data=df, x='Filter', y=name, hue='Model', palette=PALETTE, 
                             dodge=0.5, join=False, markers='D', errorbar=None)
            elif mode == 'box':
                sns.boxplot(data=df, x='Filter', y=name, hue='Model', palette=PALETTE)
            else: # bar
                sns.barplot(data=df, x='Filter', y=name, hue='Model', palette=PALETTE)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.ylabel(ylabel)
            if ylim: plt.ylim(ylim)
            if name == 'Acc': plt.axhline(0.5, color='red', linestyle='--', alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(out_dir / out_name, dpi=300)
            plt.close()

    # --- 2. Stability Plot ---
    if stability_metrics:
        stab_rows = []
        for m in stability_metrics:
            dist = m.get('R2_dist', [])
            if len(dist) == 0:
                stab_rows.append({'Model': m['Model'], 'Filter': m['Filter'], 'R2': m['R2_at_40']})
            else:
                for val in dist:
                    stab_rows.append({'Model': m['Model'], 'Filter': m['Filter'], 'R2': val})
        
        stab_df = pd.DataFrame(stab_rows)
        # Focus on 'all' situations for stability
        plot_stab = stab_df[stab_df['Filter'] == 'all']
        if not plot_stab.empty:
            plt.figure(figsize=(11, 6))
            if mode == 'violin':
                sns.violinplot(data=plot_stab, x='Model', y='R2', palette=PALETTE, inner='quartile')
                sns.pointplot(data=plot_stab, x='Model', y='R2', palette=PALETTE, 
                             join=False, markers='D', errorbar=None)
            elif mode == 'box':
                sns.boxplot(data=plot_stab, x='Model', y='R2', palette=PALETTE)
            else:
                sns.barplot(data=plot_stab, x='Model', y='R2', palette=PALETTE)
                
            plt.title('Model Stability: Reliability R² Distribution (at 40 Games)', fontsize=14, fontweight='bold')
            plt.ylabel('Reliability R²')
            plt.ylim(0, 0.6) # Tighten stability axis
            plt.tight_layout()
            plt.savefig(out_dir / 'modern_benchmark_stability.png', dpi=300)
            plt.close()

    # --- 3. Forecasting Plot ---
    if prediction_metrics:
        pred_df = build_dist_df(prediction_metrics, 'R2_dist', 'R2')
        if not pred_df.empty:
            plt.figure(figsize=(11, 6))
            if mode == 'violin':
                sns.violinplot(data=pred_df, x='Filter', y='R2', hue='Model', palette=PALETTE, inner='quartile')
                sns.pointplot(data=pred_df, x='Filter', y='R2', hue='Model', palette=PALETTE, 
                             dodge=0.4, join=False, markers='D', errorbar=None)
            elif mode == 'box':
                sns.boxplot(data=pred_df, x='Filter', y='R2', hue='Model', palette=PALETTE)
            else:
                sns.barplot(data=pred_df, x='Filter', y='R2', hue='Model', palette=PALETTE)
                
            plt.title('Forecasting Performance: Win Rate R² Distribution', fontsize=14, fontweight='bold')
            plt.ylabel('Cumulative Win Rate R²')
            plt.ylim(0, 0.45) # Tighten forecasting axis
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(out_dir / 'modern_benchmark_forecasting.png', dpi=300)
            plt.close()

    logger.info(f"Benchmark visualizations ({mode}) saved to {out_dir}")

def run_benchmarking(quick=False, plot_mode='violin', seed=42):
    """Orchestrates all benchmarking analyses."""
    n_boot = 5 if quick else 100
    n_hg_reps = 10 if quick else 500
    n_sim_reps = 2 if quick else 5
    
    # Discovery available Modern Era seasons
    all_seasons = DataUtils.get_available_seasons()
    modern_seasons = sorted([s for s in all_seasons if int(s) >= 20202021])
    
    # Ensure latest_full_season actually has data
    latest_full_season = '20232024' # Default fallback
    for s in reversed(modern_seasons):
        try:
            df_check = DataUtils.load_season_data(s)
            if len(df_check['game_id'].unique()) > 500:
                latest_full_season = s
                break
        except:
            continue
    
    logger.info(f"Starting Modern Era Benchmarking Suite | Seasons: {modern_seasons}")
    logger.info(f"Models: {MODELS} | Filters: {FILTERS} | Mode: {plot_mode}")
    
    all_metrics = []
    stability_metrics = []
    prediction_metrics = []
    
    for model_name in MODELS:
        for f in FILTERS:
            logger.info(f"\n>>>> Evaluating MODEL: {model_name} | FILTER: {f} <<<<")
            
            # 1. Initialize Evaluator
            evaluator = PredictiveEvaluator(
                model_name, 'gd', f, 
                n_boot=n_boot, 
                seed=seed, 
                no_dashboards=True,
                n_jobs=-1
            )
            
            # 2. Predictive Power Sweep
            logger.info(f"Running Predictive Power Sweep ({n_boot} reps)...")
            season_results = []
            for season in modern_seasons:
                res = evaluator.run_evaluation(season, n_reps=n_boot)
                if res:
                    season_results.append(res)
            
            if season_results:
                combined_raw = pd.concat([r['Raw_Results'] for r in season_results], ignore_index=True)
                final_metrics = evaluator.calculate_metrics(combined_raw)
                all_metrics.append({
                    'Model': model_name,
                    'Filter': f,
                    'Brier': final_metrics['Brier'],
                    'Brier_dist': final_metrics.get('Brier_dist', []),
                    'Brier_CI': f"{final_metrics['Brier_lo']:.4f} - {final_metrics['Brier_hi']:.4f}",
                    'Acc': final_metrics['Accuracy'],
                    'Acc_dist': final_metrics.get('Accuracy_dist', []),
                    'Acc_CI': f"{final_metrics['Accuracy_lo']:.3f} - {final_metrics['Accuracy_hi']:.3f}"
                })
            
            # 3. Hockey-Graphs Stability
            logger.info(f"Running Hockey-Graphs Stability ({n_hg_reps} reps)...")
            hg_df = evaluator.run_hockey_graphs_stability(modern_seasons, reps=n_hg_reps)
            target_hg = hg_df[hg_df['Sample_Size'] == 40]
            if not target_hg.empty:
                r2_col = 'Goals_r2' if model_name == 'actual' else 'xG_r2'
                dist_key = 'goal' if model_name == 'actual' else 'xg'
                # Get distribution from attrs
                hg_dist = hg_df.attrs.get('dist_map', {}).get(40, {}).get(dist_key, [])
                stability_metrics.append({
                    'Model': model_name,
                    'Filter': f,
                    'R2_at_40': target_hg[r2_col].iloc[0],
                    'R2_dist': hg_dist
                })
            
            # 4. Season Prediction R2
            if model_name != 'actual':
                logger.info(f"Running Season Prediction R2 ({n_sim_reps} split reps)...")
                pred_df = evaluator.run_season_prediction(
                    latest_full_season, 
                    train_split=0.7, 
                    split_reps=n_sim_reps
                )
                if pred_df is not None:
                    prediction_metrics.append({
                        'Model': model_name,
                        'Filter': f,
                        'Mean_R2': pred_df.attrs.get('MeanR2', 0.0),
                        'R2_dist': pred_df.attrs.get('r2_dist', [])
                    })
    
    # --- Generate Report ---
    report_path = Path("analysis/evaluation/modern_era_benchmark_results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Modern Era Benchmarking Suite Results\n\n")
        f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Seasons**: {modern_seasons}\n")
        f.write(f"**Quick Mode**: {quick}\n\n")
        
        f.write("## 1. Predictive Power (Game Outcome)\n")
        f.write("| Model | Filter | Brier Score (Avg) | 95% CI | Accuracy (Avg) | 95% CI |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m in all_metrics:
            f.write(f"| {m['Model']} | {m['Filter']} | {m['Brier']:.4f} | {m['Brier_CI']} | {m['Acc']:.3f} | {m['Acc_CI']} |\n")
            
        f.write("\n## 2. Stability (Hockey-Graphs Reliability R² at 40 Games)\n")
        f.write("| Model | Filter | R² at 40 Games |\n")
        f.write("| :--- | :--- | :--- |\n")
        for m in stability_metrics:
            f.write(f"| {m['Model']} | {m['Filter']} | {m['R2_at_40']:.4f} |\n")
            
        f.write("\n## 3. Season Prediction (Cumulative R² Stability)\n")
        f.write("| Model | Filter | Win Rate R² |\n")
        f.write("| :--- | :--- | :--- |\n")
        for m in prediction_metrics:
             f.write(f"| {m['Model']} | {m['Filter']} | {m['Mean_R2']:.4f} |\n")

    # --- Generate Plots ---
    plot_benchmark_results(all_metrics, stability_metrics, prediction_metrics, report_path.parent, mode=plot_mode)

    logger.info(f"\nBenchmarking Complete. Summary saved to {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run with minimal iterations for testing')
    parser.add_argument('--plot-mode', type=str, default='violin', choices=['violin', 'box', 'bar'], help='Type of plot to generate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_benchmarking(quick=args.quick, plot_mode=args.plot_mode, seed=args.seed)

if __name__ == "__main__":
    main()
