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

def main():
    parser = argparse.ArgumentParser(description="Plot Predictive Power Evaluation Results")
    parser.add_argument('--input', type=str, default='analysis/evaluation/predictive_power_20242025_all.csv', help='Path to results CSV')
    parser.add_argument('--output-dir', type=str, default='analysis/evaluation/', help='Directory to save plots')
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_plots(input_path, out_dir)

if __name__ == "__main__":
    main()
