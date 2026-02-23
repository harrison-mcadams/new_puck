import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.prediction import TeamPredictor

def generate_scatter_plot(df, x_col, y_col, title, output_path, team_col='TeamAbbrev'):
    """Helper to generate consistent scatter plots mirroring analyze.py/daily.py"""
    plt.figure(figsize=(10, 10))
    
    if df.empty:
        logger.warning(f"Empty dataframe for scatter {title}")
        return
        
    max_val = max(df[x_col].max(), df[y_col].max())
    min_val = min(df[x_col].min(), df[y_col].min())
    if pd.isna(max_val) or pd.isna(min_val):
        logger.warning(f"NaN limits for scatter {title}")
        return
        
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
    limit_max = max_val + padding
    limit_min = max(0, min_val - padding)
    
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)
    
    # Invert Y axis (lower GA is better)
    plt.gca().invert_yaxis()
    
    # Unity line
    plt.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', alpha=0.5, label='x=y')
    
    # League Averages
    avg_x = df[x_col].mean()
    avg_y = df[y_col].mean()
    plt.axvline(avg_x, color='k', linestyle=':', alpha=0.3, label=f'Avg For ({avg_x:.2f})')
    plt.axhline(avg_y, color='k', linestyle=':', alpha=0.3, label=f'Avg Against ({avg_y:.2f})')
    
    # Plot Points
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=team_col, palette='tab20', s=100, legend=False)
    
    # Labels
    for i, r in df.iterrows():
        plt.text(r[x_col]+(padding*0.05), r[y_col], r[team_col], fontsize=9)
        
    plt.title(title)
    plt.xlabel('Mixed Effects xtG For / 60')
    plt.ylabel('Mixed Effects xtG Against / 60')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved scatter plot to {output_path}")

def plot_intercepts_scatter(coefs_df, state, output_dir):
    """Plot Offensive Intercept vs Defensive Intercept for the given state"""
    state_df = coefs_df[coefs_df['game_state'] == state].copy()
    if state_df.empty:
        return
        
    pivot_df = state_df.pivot(index='team', columns='role', values='coef').reset_index()
    if 'Offense' not in pivot_df.columns or 'Defense' not in pivot_df.columns:
        return
        
    plt.figure(figsize=(10, 10))
    
    x_col = 'Offense'
    y_col = 'Defense'
    
    max_val = max(pivot_df[x_col].max(), pivot_df[y_col].max())
    min_val = min(pivot_df[x_col].min(), pivot_df[y_col].min())
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
    limit_max = max_val + padding
    limit_min = max(0, min_val - padding)
    
    abs_max = max(abs(limit_min), abs(limit_max))
    plt.xlim(-abs_max, abs_max)
    plt.ylim(-abs_max, abs_max)
    
    # Invert Y axis: Negative defensive intercept means FEWER goals allowed (GOOD)
    plt.gca().invert_yaxis()
    
    plt.axvline(0, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    sns.scatterplot(data=pivot_df, x=x_col, y=y_col, hue='team', palette='tab20', s=100, legend=False)
    
    for i, r in pivot_df.iterrows():
        plt.text(r[x_col]+(abs_max*0.02), r[y_col], r['team'], fontsize=9)
        
    plt.title(f'Team Isolated Talent {state} (Mixed Effects Intercepts)')
    plt.xlabel('Offensive Intercept (Higher = More Goals For)')
    plt.ylabel('Defensive Intercept (Negative = Fewer Goals Against)')
    plt.grid(True, alpha=0.3)
    
    out_path = output_dir / f'intercepts_scatter_{state}.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved intercepts plot to {out_path}")

def main():
    season = "20252026"
    base_dir = Path("analysis/xgs/mixed_effects")
    out_dir = Path("analysis/prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return
        
    # 1. Plot Intercepts
    coefs_csv = base_dir / "mixed_effects_coefficients.csv"
    if coefs_csv.exists():
        logger.info(f"Loading {coefs_csv} for intercepts plots")
        df_coefs = pd.read_csv(coefs_csv)
        for st in ['5v5', '5v4', '4v5']:
            plot_intercepts_scatter(df_coefs, st, base_dir)
    else:
        logger.warning(f"Coefficients CSV not found: {coefs_csv}")

    # 2. Predict Team Strengths & Plot Scatter
    logger.info("Initializing Team Predictor to calculate xtG For/Against...")
    pred = TeamPredictor(season=season)
    pred._load_data = pred._load_data_refined
    pred._load_data()
    
    results = []
    for tid in pred.teams_list:
        row = pred.skill_proxies[pred.skill_proxies['TeamAbbrev'] == tid]
        if row.empty: continue
        real_id = str(row.iloc[0]['TeamID'])
        
        res = pred.predict(real_id)
        if res:
            res['TeamAbbrev'] = tid
            res['TeamID'] = real_id
            results.append(res)
            
    if not results:
        logger.error("No prediction results generated.")
        return
        
    df_res = pd.DataFrame(results)
    
    # 2a. Overall Scatter
    generate_scatter_plot(
        df_res, 'GF_60', 'GA_60', 
        f'Predicted xtG Performance (All Situations) - {season}',
        out_dir / 'mixed_effects_scatter_overall.png'
    )
    
    # 2b. Per-State Scatters
    states = ['5v5', '5v4', '4v5']
    for st in states:
        st_data = []
        for r in results:
            det = r['details'].get(st)
            if not det: continue
            
            t_stats = pred.base_stats.get(r['TeamID'])
            if not t_stats: continue
            
            pace = pred.league_pace
            base_xf = t_stats['rates'].get(st, 0)
            base_xa = t_stats['def_rates'].get(st, 0)
            mult_f = det['Mult_F']
            mult_a = det['Mult_A']
            
            rate_f = base_xf * mult_f * pace
            rate_a = base_xa * mult_a * pace
            
            st_data.append({
                'TeamAbbrev': r['TeamAbbrev'],
                f'{st}_xtG_For_60': rate_f,
                f'{st}_xtG_Against_60': rate_a
            })
            
        if st_data:
            st_df = pd.DataFrame(st_data)
            generate_scatter_plot(
                st_df, f'{st}_xtG_For_60', f'{st}_xtG_Against_60',
                f'{st} Mixed Effects xtG Performance (Rate per 60m) - {season}',
                out_dir / f'mixed_effects_scatter_{st}.png'
            )

if __name__ == "__main__":
    main()
