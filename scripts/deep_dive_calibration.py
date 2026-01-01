
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgboost_nested, fit_xgs, analyze, correction, impute, config

def deep_dive_analysis():
    print("--- Deep Dive Calibration Analysis ---")
    
    # 1. Load Data (Iterate all seasons)
    base_dir = Path(config.DATA_DIR)
    seasons = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and item.name.isdigit():
             csv_path = item / f"{item.name}_df.csv"
             if csv_path.exists():
                 s_name = f"{item.name[:4]}-{item.name[4:]}"
                 seasons.append((s_name, str(csv_path)))
                 
    # Manual check for 2025
    if not any(s[0] == '2025-2026' for s in seasons):
         p = base_dir / '20252026' / '20252026_df.csv'
         if p.exists(): seasons.append(('2025-2026', str(p)))
    
    seasons = sorted(seasons, key=lambda x: x[0])
    
    # Storage
    metrics = []
    acc_by_dist = []
    
    print(f"Processing {len(seasons)} seasons...")
    
    for s_name, s_path in seasons:
        print(f"  Loading {s_name}...")
        df = pd.read_csv(s_path)
        
        # Pipeline (Correction -> Imputation)
        # correction
        if 'event' in df.columns:
             # Fast check if correction needed (team_id logic hard to check without load context, 
             # but safe to run if it wasn't run. If loaded from processed cache it might be done?
             # load_season_df DOES NOT run it. So typically raw CSV is raw.
             df = correction.fix_blocked_shot_attribution(df)
        
        # impute
        mask_shots = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])
        df_shots = df[mask_shots].copy()
        
        # Try impute
        try:
             df_shots = impute.impute_blocked_shot_origins(df_shots, method='point_pull')
        except: pass
        
        # --- METRICS ---
        
        # 1. Unblocked Shot Characteristics
        mask_ub = df_shots['event'] != 'blocked-shot'
        df_ub = df_shots[mask_ub].copy()
        
        mean_dist = df_ub['distance'].mean()
        mean_angle = df_ub['angle_deg'].abs().mean()
        
        # 2. Block Rate
        n_total = len(df_shots)
        n_block = (df_shots['event'] == 'blocked-shot').sum()
        block_rate = n_block / n_total if n_total > 0 else 0
        
        # 3. Accuracy (On Net / Unblocked)
        # On Net = SOG + Goal
        mask_on_net = df_ub['event'].isin(['shot-on-goal', 'goal'])
        acc_rate = mask_on_net.mean()
        
        # 4. Shooting % (Goal / SOG+Goal)
        # Shooting Percentage is typically goals / shots on goal
        df_on_net_rows = df_ub[mask_on_net]
        goals = (df_on_net_rows['event'] == 'goal').sum()
        sots = len(df_on_net_rows)
        sh_pct = goals / sots if sots > 0 else 0
        
        # 5. Goal Rate (Goal / Unblocked Attempt) aka Fenwick Shooting %
        goal_rate_fenwick = goals / len(df_ub) if len(df_ub) > 0 else 0
        
        metrics.append({
            'Season': s_name,
            'Mean Distance': mean_dist,
            'Mean Angle': mean_angle,
            'Block Rate': block_rate,
            'Accuracy Rate': acc_rate,
            'Shooting %': sh_pct,
            'Fenwick S%': goal_rate_fenwick,
            'Count': n_total
        })
        
        # --- bin analysis ---
        # Accuracy over distance bins
        df_ub['dist_bin'] = pd.cut(df_ub['distance'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
        bin_acc = df_ub.groupby('dist_bin')['event'].apply(lambda x: x.isin(['shot-on-goal', 'goal']).mean())
        for b, v in bin_acc.items():
            acc_by_dist.append({'Season': s_name, 'Bin': b, 'Accuracy': v})

    # Save Results
    df_metrics = pd.DataFrame(metrics)
    df_bins = pd.DataFrame(acc_by_dist)
    
    df_metrics.to_csv('analysis/deep_dive_metrics.csv', index=False)
    df_bins.to_csv('analysis/deep_dive_bins.csv', index=False)
    
    # --- PLOTS ---
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Shot Behavior (Distance)
    ax1 = plt.subplot(2, 3, 1)
    sns.lineplot(data=df_metrics, x='Season', y='Mean Distance', marker='o', ax=ax1, color='tab:blue')
    ax1.set_title('Mean Shot Distance (Unblocked)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Block Rate
    ax2 = plt.subplot(2, 3, 2)
    sns.lineplot(data=df_metrics, x='Season', y='Block Rate', marker='o', ax=ax2, color='tab:orange')
    ax2.set_title('Block Rate (% of Attempts)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Rate (The Culprit?)
    ax3 = plt.subplot(2, 3, 3)
    sns.lineplot(data=df_metrics, x='Season', y='Accuracy Rate', marker='o', ax=ax3, color='tab:green')
    ax3.set_title('Accuracy Rate (% On Net | Unblocked)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Shooting %
    ax4 = plt.subplot(2, 3, 4)
    sns.lineplot(data=df_metrics, x='Season', y='Shooting %', marker='o', ax=ax4, color='tab:red')
    ax4.set_title('Shooting % (Goals / SOG)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy by Distance Bin (Heatmap or Multi-line)
    # Pivot df_bins
    ax5 = plt.subplot(2, 3, 5)
    pivot_bins = df_bins.pivot(index='Season', columns='Bin', values='Accuracy')
    sns.heatmap(pivot_bins, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax5)
    ax5.set_title('Accuracy by Distance Bin')
    
    # 6. Distance vs Accuracy Trend (Changes relative to 2014)
    # Normalized to 2014 baseline for clarity?
    # Or just mean angle
    ax6 = plt.subplot(2, 3, 6)
    sns.lineplot(data=df_metrics, x='Season', y='Mean Angle', marker='o', ax=ax6, color='tab:purple')
    ax6.set_title('Mean Shot Angle (Abs Degrees)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/deep_dive_trends.png')
    print("Deep Dive Complete. Plots saved to analysis/deep_dive_trends.png")

if __name__ == "__main__":
    deep_dive_analysis()
