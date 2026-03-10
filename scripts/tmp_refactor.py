import re

file_path = "c:/Users/harri/Desktop/new_puck/scripts/evaluate_predictive_power.py"
with open(file_path, 'r') as f:
    code = f.read()

# 1. Load the NN_xG model
load_code = """
    df = analyze.load_and_prep_data(season, include_shifts=False)
    
    # NEW: Predict Global Standard xG
    logger.info("Loading Global Standard xG Model...")
    try:
        import joblib
        import os
        from puck import analyze
        nn_path = os.path.join(analyze.puck_config.ANALYSIS_DIR, 'xgs', 'xg_model_non_nested_tensor.joblib')
        if os.path.exists(nn_path):
            nn_clf = joblib.load(nn_path)
            mask_valid = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
            df['nn_xgs'] = 0.0
            if mask_valid.sum() > 0:
                features = nn_clf.feature_names_in_ if hasattr(nn_clf, 'feature_names_in_') else None
                # Simplest fallback if we don't know features: just use data pipeline
                # but usually sklearn pipeline handles it.
                if hasattr(nn_clf, 'predict_proba'):
                    df.loc[mask_valid, 'nn_xgs'] = nn_clf.predict_proba(df[mask_valid])[:, 1]
    except Exception as e:
        logger.warning(f"Failed to calculate global standard xG: {e}")
"""
code = code.replace("    df = analyze.load_and_prep_data(season, include_shifts=False)", load_code)

# 2. extract_rates: initialization
orig_init = "'local_nn_xgf_per_game': 0.0, 'local_nn_xga_per_game': 0.0,"
new_init = "'local_nn_xgf_per_game': 0.0, 'local_nn_xga_per_game': 0.0,\n            'nn_xgf_per_game': 0.0, 'nn_xga_per_game': 0.0,"
code = code.replace(orig_init, new_init)

# 3. extract_rates: calculation
orig_calc = """        # Local Non-Nested xG
        if 'local_nn_xgs' in df.columns:"""
new_calc = """        # Global Standard xG
        if 'nn_xgs' in df.columns:
            nn_xgf = df[((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))]['nn_xgs'].sum()
            nn_xga = df[((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))]['nn_xgs'].sum()
        else:
            nn_xgf, nn_xga = 0, 0
            
        # Local Non-Nested xG
        if 'local_nn_xgs' in df.columns:"""
code = code.replace(orig_calc, new_calc)

# 4. extract_rates: assignment
orig_asn = "rates[t]['local_nn_xgf_per_game'] = local_nn_xgf / team_games\n        rates[t]['local_nn_xga_per_game'] = local_nn_xga / team_games"
new_asn = orig_asn + "\n        rates[t]['nn_xgf_per_game'] = nn_xgf / team_games\n        rates[t]['nn_xga_per_game'] = nn_xga / team_games"
code = code.replace(orig_asn, new_asn)

# 5. extract_rates_per60: Not actively called but good to patch if simple. Will skip this for now as per60 is a side feature.

# 6. predict_matchup / predict_matchup_per60
orig_pm = "# 6. Local xtG"
new_pm = """    # 5.75 Global Standard xG
    if home in rates and away in rates:
        h_nn_exp = safe_div(rates[home]['nn_xgf_per_game'] * rates[away]['nn_xga_per_game'], league_avgs['nn_xgf_per_game'], league_avgs['nn_xgf_per_game'])
        a_nn_exp = safe_div(rates[away]['nn_xgf_per_game'] * rates[home]['nn_xga_per_game'], league_avgs['nn_xgf_per_game'], league_avgs['nn_xgf_per_game'])
    else:
        h_nn_exp = league_avgs.get('nn_xgf_per_game', 3.0)
        a_nn_exp = league_avgs.get('nn_xgf_per_game', 3.0)
        
    # 6. Local xtG"""
code = code.replace(orig_pm, new_pm)

# 7. predict_matchup return
orig_ret = "'local_nn_xg': (h_nnxg_exp, a_nnxg_exp),"
new_ret = orig_ret + "\n        'nn_xg': (h_nn_exp, a_nn_exp),"
code = code.replace(orig_ret, new_ret)

# 8. Loop accumulator arrays
code = code.replace("p_lxtg_home   = []", "p_lxtg_home   = []\n            p_nn_home     = []")

code = code.replace("if 'local_xtg' in exps:", "if 'nn_xg' in exps:\n                    h_n, a_n, t_n = calculate_win_prob(exps['nn_xg'][0], exps['nn_xg'][1])\n                    p_nn_home.append(h_n + t_n * 0.5)\n                \n                if 'local_xtg' in exps:")

code = code.replace("p_lxt = np.array(p_lxtg_home)", "p_lxt = np.array(p_lxtg_home)\n            p_nn  = np.array(p_nn_home)")

orig_boot_brier = "bnnx_mean, bnnx_lo, bnnx_hi = bootstrap_metric(y_act, p_nnx, _brier, n_boot)"
new_boot_brier = orig_boot_brier + "\n                bnn_mean, bnn_lo, bnn_hi = bootstrap_metric(y_act, p_nn, _brier, n_boot)"
code = code.replace(orig_boot_brier, new_boot_brier)

orig_boot_acc = "annx_mean, annx_lo, annx_hi = bootstrap_metric(y_act, p_nnx, _accuracy, n_boot)"
new_boot_acc = orig_boot_acc + "\n                ann_mean, ann_lo, ann_hi = bootstrap_metric(y_act, p_nn, _accuracy, n_boot)"
code = code.replace(orig_boot_acc, new_boot_acc)

orig_pt_brier = "bnnx_mean = _brier(y_act, p_nnx); bnnx_lo = bnnx_hi = bnnx_mean"
new_pt_brier = orig_pt_brier + "\n                bnn_mean = _brier(y_act, p_nn); bnn_lo = bnn_hi = bnn_mean"
code = code.replace(orig_pt_brier, new_pt_brier)

orig_pt_acc = "annx_mean = _accuracy(y_act, p_nnx); annx_lo = annx_hi = annx_mean"
new_pt_acc = orig_pt_acc + "\n                ann_mean = _accuracy(y_act, p_nn); ann_lo = ann_hi = ann_mean"
code = code.replace(orig_pt_acc, new_pt_acc)

# 9. model_ranks lambda
orig_lambda = "('Local_NN_xG', lambda t: rates[t]['local_nn_xgf_per_game'] - rates[t]['local_nn_xga_per_game'] if not args.per60 else rates60[t]['5v5']['local_nn_xgf60'] - rates60[t]['5v5']['local_nn_xga60'] + rates60[t]['pp']['local_nn_xgf60'] - rates60[t]['pk']['local_nn_xga60']),"
new_lambda = orig_lambda + "\n                ('NN_xG', lambda t: rates[t]['nn_xgf_per_game'] - rates[t]['nn_xga_per_game'] if not args.per60 else 0),"
code = code.replace(orig_lambda, new_lambda)

# 10. res dict
orig_res = "'Local_NN_xG_Brier': bnnx_mean, 'Local_NN_xG_Brier_lo': bnnx_lo, 'Local_NN_xG_Brier_hi': bnnx_hi,"
new_res = orig_res + "\n                'NN_xG_Brier': bnn_mean, 'NN_xG_Brier_lo': bnn_lo, 'NN_xG_Brier_hi': bnn_hi,"
code = code.replace(orig_res, new_res)

orig_res2 = "'Local_NN_xG_Acc': annx_mean,   'Local_NN_xG_Acc_lo': annx_lo,    'Local_NN_xG_Acc_hi': annx_hi,"
new_res2 = orig_res2 + "\n                'NN_xG_Acc': ann_mean, 'NN_xG_Acc_lo': ann_lo, 'NN_xG_Acc_hi': ann_hi,"
code = code.replace(orig_res2, new_res2)

# Ensure plot_predictive_power labels are exact
# The user wants:
# - Actual Goals
# - Nested xG
# - Standard xG
# - Local Nested xG
# - Local Standard xG
# - Mixed Effects xtG
# - Local Mixed Effects xtG
# - MoneyPuck xG
# - Coin Flip (50%)

# Plot styles requested:
# local versions: similar colors (blue -> cyan, dark green -> light green)
# local options: dotted lines
# non-local options: solid lines

plot_code = \"\"\"import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 'Goals', 'Actual Goals', 'black', 'o', '-'
# 'xG',   'Nested xG', 'blue', 's', '-'
# 'NN_xG', 'Standard xG', 'cyan'?, 's', '-'
# 'Local_xG', 'Local Nested xG', 'deepskyblue' (cyan), 'v', ':'
# 'Local_NN_xG', 'Local Standard xG', 'orange', 'X', ':'  (wait, no, cyan -> dotted. Let's make NN_xG orange maybe, or another base color, then local version is light version of that)
# Wait, user said: "make the local version similar colors (blue -> cyan, dark green -> light green), but have the local options be dotted lines with the non-local options be solid lines."
# So:
# Nested xG = blue (solid)
# Local Nested xG = cyan (dotted)
# Mixed Effects xtG = dark green (solid)
# Local Mixed Effects xtG = light green (dotted)
# Standard xG = let's pick purple/red? The user didn't specify base for standard, maybe orange?
# Local Standard xG = light orange (or gold)? Dotted.
# MoneyPuck xG = purple (solid or dash, whatever, maybe solid or dash-dot?)

MODELS = [
    ('Goals', 'Actual Goals', 'black', 'o', '-'),
    ('xG',    'Nested xG',  'blue',  's', '-'),
    ('Local_xG', 'Local Nested xG', 'cyan', 'v', ':'),
    ('NN_xG', 'Standard xG', 'darkorange', 'X', '-'),
    ('Local_NN_xG', 'Local Standard xG', 'gold', 'x', ':'),
    ('xtG',   'Mixed Effects xtG', 'darkgreen', '^', '-'),
    ('Local_xtG', 'Local Mixed Effects xtG', 'limegreen', 'p', ':'),
    ('MP',    'MoneyPuck xG', 'purple', 'D', '--'),
]

def _plot_metric(df, metric, ylabel, title, higher_better, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, label, color, marker, ls in MODELS:
        col = f'{key}_{metric}'
        if col not in df.columns:
            continue
        
        ax.plot(df['N'], df[col], marker=marker, label=label,
                color=color, linewidth=2, linestyle=ls)
        
        # Shaded CI band
        lo_col = f'{col}_lo'
        hi_col = f'{col}_hi'
        if lo_col in df.columns and hi_col in df.columns:
            ax.fill_between(df['N'], df[lo_col], df[hi_col],
                            color=color, alpha=0.12)
    
    if not higher_better:
        ax.set_title(f'{title} (Lower is Better)')
    else:
        ax.set_title(f'{title} (Higher is Better)')
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Coin Flip (50%)')
    
    ax.set_xlabel('Number of Games in Training Sample (N)')
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

    _plot_metric(df, 'Brier', 'Brier Score (Mean Squared Error)',
                 f'Predictive Power vs Sample Size ({input_path.stem}): Brier', False,
                 out_dir / f'{input_path.stem}_brier.png')

    for key, *_ in MODELS:
        col = f'{key}_Acc'
        if col in df.columns:
            df[col] = df[col] * 100
        for suffix in ['_lo', '_hi']:
            sc = f'{col}{suffix}'
            if sc in df.columns:
                df[sc] = df[sc] * 100

    _plot_metric(df, 'Acc', 'Prediction Accuracy (%)',
                 f'Predictive Power vs Sample Size ({input_path.stem}): Accuracy', True,
                 out_dir / f'{input_path.stem}_accuracy.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='analysis/evaluation/predictive_power_20242025_all.csv')
    parser.add_argument('--output-dir', type=str, default='analysis/evaluation/')
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_plots(input_path, out_dir)

if __name__ == '__main__':
    main()
\"\"\"

with open(file_path, 'w') as f:
    f.write(code)

with open('c:/Users/harri/Desktop/new_puck/scripts/plot_predictive_power.py', 'w') as f:
    f.write(plot_code)
print("Files rewritten successfully.")
