import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

MODELS = [
    ('Goals', 'Actual Goals', 'black', 'o', '-'),
    ('xG',    'Nested xG',  'blue',  's', '-'),
    ('NN_xG', 'Standard xG', 'darkorange', 'X', '-'),
    ('Local_xG', 'Local Nested xG', 'cyan', 'v', ':'),
    ('Local_NN_xG', 'Local Standard xG', 'gold', 'x', ':'),
    ('xtG',   'Mixed Effects xtG', 'darkgreen', '^', '-'),
    ('Local_xtG', 'Local Mixed Effects xtG', 'limegreen', 'p', ':'),
    ('MP',    'MoneyPuck xG', 'purple', 'D', '-'),
]

def _plot_metric(df, metric, ylabel, title, higher_better, out_path):
    """Generic plotter that handles optional CI columns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, label, color, marker, ls in MODELS:
        col = f'{key}_{metric}'
        if col not in df.columns:
            continue
        
        ax.plot(df['N'], df[col], marker=marker, label=label,
                color=color, linewidth=2, linestyle=ls)
        
        # Shaded CI band (if columns present)
        lo_col = f'{col}_lo'
        hi_col = f'{col}_hi'
        if lo_col in df.columns and hi_col in df.columns:
            ax.fill_between(df['N'], df[lo_col], df[hi_col],
                            color=color, alpha=0.12)
    
    if not higher_better:
        ax.set_title(f'{title} (Lower is Better)')
    else:
        ax.set_title(f'{title} (Higher is Better)')
        if metric == 'Acc':
            ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Coin Flip (50%)')
    
    ax.set_xlabel('Number of Games in Training Sample (N)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_plots(input_path: Path, out_dir: Path):
    if not input_path.exists():
        print(f"Error: Could not find input file {input_path}")
        return

    df = pd.read_csv(input_path)

    sns.set_theme(style="whitegrid")

    # Brier Score
    _plot_metric(df, 'Brier', 'Brier Score (Mean Squared Error)',
                 f'Predictive Power vs Sample Size ({input_path.stem}): Brier', False,
                 out_dir / f'{input_path.stem}_brier.png')

    # Accuracy (convert to percentage)
    accuracy_df = df.copy()
    for key, *_ in MODELS:
        col = f'{key}_Acc'
        if col in accuracy_df.columns:
            accuracy_df[col] = accuracy_df[col] * 100
        for suffix in ['_lo', '_hi']:
            sc = f'{col}{suffix}'
            if sc in accuracy_df.columns:
                accuracy_df[sc] = accuracy_df[sc] * 100

    _plot_metric(accuracy_df, 'Acc', 'Prediction Accuracy (%)',
                 f'Predictive Power vs Sample Size ({input_path.stem}): Accuracy', True,
                 out_dir / f'{input_path.stem}_accuracy.png')

    # Spearman EOS
    _plot_metric(df, 'Spearman_EOS', 'Spearman Correlation Coefficient',
                 f'Rank Correlation (EOS) vs Sample Size ({input_path.stem})', True,
                 out_dir / f'{input_path.stem}_spearman_eos.png')

    # Spearman ROS
    _plot_metric(df, 'Spearman_ROS', 'Spearman Correlation Coefficient',
                 f'Rank Correlation (ROS) vs Sample Size ({input_path.stem})', True,
                 out_dir / f'{input_path.stem}_spearman_ros.png')

    print(f"Successfully saved evaluation plots to {out_dir}")

def plot_accuracy_vs_parity(results_csv: Path, parity_csv: Path, out_path: Path):
    """Plots Model Accuracy vs League Parity (Std Dev of Points %) over time."""
    if not results_csv.exists() or not parity_csv.exists():
        print(f"Skipping parity plot: missing {results_csv} or {parity_csv}")
        return

    res_df = pd.read_csv(results_csv)
    par_df = pd.read_csv(parity_csv)

    # Filter for Nested xG Global model
    model_df = res_df[(res_df['Model'] == 'nested_xg') & (res_df['Filter'] == 'all')].copy()
    
    # Merge on Season
    # Ensure season formats match (CSV has 20202021, Parity has 20202021)
    model_df['Season'] = model_df['Season'].astype(str)
    par_df['season'] = par_df['season'].astype(str)
    
    merged = pd.merge(model_df, par_df, left_on='Season', right_on='season')
    merged = merged.sort_values('Season')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Model Accuracy', color=color)
    ax1.plot(merged['Season'].values, merged['Accuracy'].values, marker='o', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.5, 0.65)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('League Parity (Points % StdDev)', color=color)
    # Lower StdDev = Higher Parity. 
    ax2.plot(merged['Season'].values, merged['points_pct_std'].values, marker='s', color=color, linewidth=2, linestyle='--', label='Parity (StdDev)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Invert ax2 if we want to show Parity increasing
    # ax2.invert_yaxis() 

    plt.title('Model Accuracy vs League Parity Trends')
    fig.tight_layout()
    plt.grid(alpha=0.2)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved accuracy vs parity plot to {out_path}")

def plot_local_delta(results_csv: Path, out_path: Path):
    """Plots the Accuracy Delta between Local and Global models by season."""
    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    
    # Filter for Nested xG Global and Local
    global_df = df[(df['Model'] == 'nested_xg') & (df['Filter'] == 'all')].copy()
    local_df = df[(df['Model'] == 'local_nested_xg') & (df['Filter'] == 'all')].copy()
    
    if global_df.empty or local_df.empty:
        print("Missing global or local models for delta plot.")
        return

    merged = pd.merge(
        global_df[['Season', 'Accuracy']], 
        local_df[['Season', 'Accuracy']], 
        on='Season', suffixes=('_global', '_local')
    )
    merged['delta'] = merged['Accuracy_local'] - merged['Accuracy_global']
    merged = merged.sort_values('Season')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tab:green' if d > 0 else 'tab:red' for d in merged['delta']]
    
    bars = ax.bar(merged['Season'].values.astype(str), merged['delta'].values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    
    # Add labels
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:+.1%}', 
                va='bottom' if yval > 0 else 'top', ha='center', fontweight='bold')

    ax.set_ylabel('Accuracy Improvement (Local - Global)')
    ax.set_title('Value of Seasonal Retraining (Accuracy Delta)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved local delta plot to {out_path}")

def plot_arena_impact(adj_results: Path, raw_results: Path, out_path: Path):
    """Compares Arena Adjusted vs Raw Tracking (No Adjustments) accuracy."""
    if not adj_results.exists() or not raw_results.exists():
        return

    adj_df = pd.read_csv(adj_results)
    raw_df = pd.read_csv(raw_results)

    # Filter for Nested xG Global
    adj_m = adj_df[(adj_df['Model'] == 'nested_xg') & (adj_df['Filter'] == 'all')].copy()
    raw_m = raw_df[(raw_df['Model'] == 'nested_xg') & (raw_df['Filter'] == 'all')].copy()

    if adj_m.empty or raw_m.empty:
        return

    merged = pd.merge(
        adj_m[['Season', 'Accuracy']], 
        raw_m[['Season', 'Accuracy']], 
        on='Season', suffixes=('_adj', '_raw')
    )
    merged = merged.sort_values('Season')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(merged['Season'].values.astype(str), merged['Accuracy_adj'].to_numpy() * 100, 
            marker='o', label='Arena Adjusted', color='blue', linewidth=2)
    ax.plot(merged['Season'].values.astype(str), merged['Accuracy_raw'].to_numpy() * 100, 
            marker='s', label='Raw Tracking (No Adj)', color='red', linewidth=2, linestyle='--')
    
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_xlabel('Season')
    ax.set_title('Impact of Arena Bias Adjustments over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved arena impact plot to {out_path}")

def plot_situational_stability(all_results: Path, sit_results: Path, out_path: Path):
    """Compares All Situations vs 5v5 accuracy."""
    if not all_results.exists() or not sit_results.exists():
        return

    all_df = pd.read_csv(all_results)
    sit_df = pd.read_csv(sit_results)

    # Filter for Nested xG Global
    all_m = all_df[(all_df['Model'] == 'nested_xg') & (all_df['Filter'] == 'all')].copy()
    sit_m = sit_df[(sit_df['Model'] == 'nested_xg')].copy() 

    if all_m.empty or sit_m.empty:
        return

    merged = pd.merge(
        all_m[['Season', 'Accuracy']], 
        sit_m[['Season', 'Accuracy']], 
        on='Season', suffixes=('_all', '_sit')
    )
    merged = merged.sort_values('Season')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(merged['Season'].values.astype(str), merged['Accuracy_all'].to_numpy() * 100, 
            marker='o', label='All Situations', color='black', linewidth=2)
    ax.plot(merged['Season'].values.astype(str), merged['Accuracy_sit'].to_numpy() * 100, 
            marker='s', label='5v5 Only', color='green', linewidth=2, linestyle='--')
    
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_xlabel('Season')
    ax.set_title('Situational Stability: All Situations vs 5v5')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved situational stability plot to {out_path}")

def plot_situational_heatmap(all_csv: Path, sit_csv_map: dict, out_path: Path):
    """Generates a Season vs Game State heatmap of Prediction Accuracy."""
    data = []
    
    # helper to load a CSV and filter for global nested_xg
    def load_filter(path, label):
        if not path.exists(): return
        df = pd.read_csv(path)
        m = df[(df['Model'] == 'nested_xg')].copy()
        for _, row in m.iterrows():
            if row['Season'] == 'Combined': continue
            data.append({'Season': str(row['Season']), 'Situation': label, 'Accuracy': row['Accuracy'] * 100})

    load_filter(all_csv, 'All')
    for label, path in sit_csv_map.items():
        load_filter(path, label)
            
    if not data:
        return
        
    df = pd.DataFrame(data)
    pivot = df.pivot(index='Season', columns='Situation', values='Accuracy')
    
    # Sort columns logically
    cols = ['All', '5v5', '5v5_close']
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=60, cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Prediction Accuracy Heatmap: Season vs Situation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved situational heatmap to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Predictive Power Evaluation Results")
    parser.add_argument('--input', type=str, default='analysis/evaluation/predictive_power_comparison_suite_all.csv', help='Path to results CSV')
    parser.add_argument('--no-adj-input', type=str, default=None, help='Path to results CSV without arena adjustments')
    parser.add_argument('--sit-input', type=str, default=None, help='Path to results CSV with situational filter (e.g. 5v5)')
    parser.add_argument('--parity', type=str, default='analysis/parity_metrics.csv', help='Path to parity CSV')
    parser.add_argument('--output-dir', type=str, default='analysis/evaluation/', help='Directory to save plots')
    args = parser.parse_args()

    input_path = Path(args.input)
    parity_path = Path(args.parity)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        generate_plots(input_path, out_dir)
        plot_accuracy_vs_parity(input_path, parity_path, out_dir / 'accuracy_vs_parity.png')
        plot_local_delta(input_path, out_dir / 'local_vs_global_delta.png')
        
        if args.no_adj_input:
            plot_arena_impact(input_path, Path(args.no_adj_input), out_dir / 'arena_adjustment_impact.png')
            
        if args.sit_input:
            plot_situational_stability(input_path, Path(args.sit_input), out_dir / 'situational_stability.png')
            
        # Try to find common situational CSVs for heatmap
        sit_map = {
            '5v5': out_dir / 'predictive_power_comparison_suite_5v5.csv',
            '5v5_close': out_dir / 'predictive_power_comparison_suite_5v5_close.csv'
        }
        plot_situational_heatmap(input_path, sit_map, out_dir / 'situational_heatmap.png')

if __name__ == "__main__":
    main()
