
import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils, ModelRegistry, TeamAbilitySummarizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_bootstrapped_logistic_bypass(n_boot=100, seed=42):
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025']
    models_to_test = ['nested_xg', 'actual']
    
    rng = np.random.default_rng(seed)
    all_rep_results = []
    
    for season in seasons:
        print(f"\nProcessing season: {season}")
        try:
            df = DataUtils.load_season_data(season)
            sched_df = DataUtils.process_schedule(df)
            all_gids = sched_df['game_id'].values
            n_total = len(all_gids)
            
            # Proportions: 50% Abilities, 20% Logistic Train, 30% Evaluation
            n_abilities = int(n_total * 0.5)
            n_log_train = int(n_total * 0.2)
            
            for r in range(n_boot):
                if r % 10 == 0:
                    print(f"  Rep {r}/{n_boot}...")
                
                # Random Shuffle
                shuffled_gids = rng.permutation(all_gids)
                
                gids_abilities = shuffled_gids[:n_abilities]
                gids_log_train = shuffled_gids[n_abilities:n_abilities + n_log_train]
                gids_test = shuffled_gids[n_abilities + n_log_train:]
                
                df_abilities = df[df['game_id'].isin(gids_abilities)]
                sched_log_train = sched_df[sched_df['game_id'].isin(gids_log_train)]
                sched_test = sched_df[sched_df['game_id'].isin(gids_test)]
                
                for m_name in models_to_test:
                    evaluator = PredictiveEvaluator(model_name=m_name, metric_type='gd', filter_type='all', n_boot=0)
                    
                    # 1. Extract abilities
                    abilities = evaluator.summarizer.get_team_abilities(df_abilities, m_name, evaluator.model_registry)
                    lg_avg = np.mean([v['for'] for v in abilities.values()])
                    
                    # --- POISSON MATCHUP ---
                    rep_p_results = []
                    for _, row in sched_test.iterrows():
                        p_hw = evaluator.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, lg_avg, evaluator.outcome_type)
                        # Use strictly binary winner target
                        actual = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0
                        rep_p_results.append({'p': p_hw, 'y': actual})
                    
                    p_res_df = pd.DataFrame(rep_p_results)
                    p_brier = brier_score_loss(p_res_df['y'], p_res_df['p'])
                    p_acc = accuracy_score(p_res_df['y'], (p_res_df['p'] >= 0.5).astype(float))
                    
                    # --- LOGISTIC MATCHUP ---
                    def get_features(sched):
                        feats = []
                        ys = []
                        for _, row in sched.iterrows():
                            h, a = row['home_team'], row['away_team']
                            if h in abilities and a in abilities:
                                h_diff = abilities[h]['for'] - abilities[h]['ag']
                                a_diff = abilities[a]['for'] - abilities[a]['ag']
                                feats.append([h_diff - a_diff])
                                target = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0
                                ys.append(target)
                        return np.array(feats), np.array(ys)
                    
                    X_train, y_train = get_features(sched_log_train)
                    X_test, y_test = get_features(sched_test)
                    
                    if len(y_train) > 1 and len(np.unique(y_train)) > 1:
                        clf = LogisticRegression()
                        clf.fit(X_train, y_train)
                        p_log = clf.predict_proba(X_test)[:, 1]
                        l_brier = brier_score_loss(y_test, p_log)
                        l_acc = accuracy_score(y_test, (p_log >= 0.5).astype(float))
                    else:
                        l_brier, l_acc = np.nan, np.nan
                        
                    all_rep_results.append({
                        'Season': season,
                        'Rep': r,
                        'Model': m_name,
                        'Poisson_Brier': p_brier,
                        'Poisson_Acc': p_acc,
                        'Logistic_Brier': l_brier,
                        'Logistic_Acc': l_acc
                    })
        except Exception as e:
            print(f"Error in {season}: {e}")

    results_df = pd.DataFrame(all_rep_results)
    
    # Aggregation
    summary = results_df.groupby(['Season', 'Model']).agg({
        'Poisson_Brier': ['mean', 'std'],
        'Poisson_Acc': ['mean', 'std'],
        'Logistic_Brier': ['mean', 'std'],
        'Logistic_Acc': ['mean', 'std']
    }).reset_index()
    
    print("\n--- Bootstrapped Results Summary (Mean +/- Std) ---")
    print(summary.to_string())
    
    os.makedirs('analysis/evaluation', exist_ok=True)
    results_df.to_csv('analysis/evaluation/logistic_bypass_bootstrapped_raw.csv', index=False)
    summary.to_csv('analysis/evaluation/logistic_bypass_bootstrapped_summary.csv', index=False)
    
    # --- VISUALIZATION ---
    plot_df = results_df.melt(id_vars=['Season', 'Model', 'Rep'], 
                              value_vars=['Poisson_Acc', 'Logistic_Acc'],
                              var_name='Engine', value_name='Accuracy')
    
    # Correct Group Labeling
    plot_df['Group'] = plot_df['Model'].astype(str) + "_" + plot_df['Engine'].astype(str).str.replace('_Acc', '')
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=plot_df, x='Season', y='Accuracy', hue='Group')
    plt.title(f"Predictive Accuracy Distribution (N={n_boot} Bootstraps)\nRandom Split")
    plt.ylim(0.45, 0.7)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analysis/evaluation/logistic_bypass_bootstrap_boxplot.png', dpi=150)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x='Season', y='Accuracy', hue='Group', capsize=.1, errorbar='se')
    plt.title(f"Mean Predictive Accuracy (N={n_boot} Bootstraps)\nPoisson vs Logistic Engine")
    plt.ylim(0.5, 0.65)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analysis/evaluation/logistic_bypass_bootstrap_barplot.png', dpi=150)
    
    print("\nVisualizations saved to analysis/evaluation/logistic_bypass_bootstrap_*.png")

if __name__ == "__main__":
    run_bootstrapped_logistic_bypass(n_boot=100)
