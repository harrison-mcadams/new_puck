
import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils, TeamAbilitySummarizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class CleanAbilitySummarizer(TeamAbilitySummarizer):
    def get_team_abilities(self, df, model_name, model_registry, strip_eng=False):
        if model_name == 'actual' and strip_eng:
            # Drop Empty Net (6v5, 5v6, etc.)
            states_to_drop = ['6v5', '5v6', '6v4', '4v6', '6v3', '3v6']
            state_col = 'relative_game_state' if 'relative_game_state' in df.columns else 'game_state'
            
            if state_col in df.columns:
                df = df[~df[state_col].isin(states_to_drop)].copy()
            
            # Drop Blowouts (> 3 goal diff)
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df = df[abs(df['home_score'] - df['away_score']) <= 3].copy()
            
        return super().get_team_abilities(df, model_name, model_registry)

def run_bootstrapped_eng_test(n_boot=100, seed=42):
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025']
    rng = np.random.default_rng(seed)
    
    summarizer = CleanAbilitySummarizer()
    all_results = []
    
    for season in seasons:
        print(f"\nProcessing season: {season}")
        try:
            df = DataUtils.load_season_data(season)
            sched_df = DataUtils.process_schedule(df)
            all_gids = sched_df['game_id'].values
            n_total = len(all_gids)
            n_train = n_total // 2
            
            for r in range(n_boot):
                if r % 25 == 0:
                    print(f"  Rep {r}/{n_boot}...")
                
                # Random Shuffle
                shuffled_gids = rng.permutation(all_gids)
                train_gids = shuffled_gids[:n_train]
                test_gids = shuffled_gids[n_train:]
                
                train_df = df[df['game_id'].isin(train_gids)]
                test_sched = sched_df[sched_df['game_id'].isin(test_gids)]
                
                # Setup models
                eval_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0)
                eval_xg = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
                
                # 1. Base Actual
                abs_base = eval_act.summarizer.get_team_abilities(train_df, 'actual', eval_act.model_registry)
                lg_avg_base = np.mean([v['for'] for v in abs_base.values()])
                
                # 2. Cleaned Actual
                abs_clean = summarizer.get_team_abilities(train_df, 'actual', eval_act.model_registry, strip_eng=True)
                lg_avg_clean = np.mean([v['for'] for v in abs_clean.values()])
                
                # 3. Base xG
                abs_xg = eval_xg.summarizer.get_team_abilities(train_df, 'nested_xg', eval_xg.model_registry)
                lg_avg_xg = np.mean([v['for'] for v in abs_xg.values()])
                
                # Predict Winners
                def get_acc(engine, abilities, avg):
                    correct = 0
                    for _, row in test_sched.iterrows():
                        p = engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, avg, 'winner')
                        actual = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0
                        if (p >= 0.5 and actual == 1.0) or (p < 0.5 and actual == 0.0):
                            correct += 1
                    return correct / len(test_sched)

                acc_base = get_acc(eval_act.matchup_engine, abs_base, lg_avg_base)
                acc_clean = get_acc(eval_act.matchup_engine, abs_clean, lg_avg_clean)
                acc_xg = get_acc(eval_xg.matchup_engine, abs_xg, lg_avg_xg)
                
                all_results.append({
                    'Season': season,
                    'Rep': r,
                    'Actual_Base': acc_base,
                    'Actual_Clean': acc_clean,
                    'nested_xg': acc_xg
                })
        except Exception as e:
            print(f"Error in {season}: {e}")

    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('Season').agg(['mean', 'std']).reset_index()
    print("\n--- Bootstrapped ENG Removal Results ---")
    print(summary.to_string())
    
    os.makedirs('analysis/evaluation', exist_ok=True)
    results_df.to_csv('analysis/evaluation/eng_removal_bootstrapped_raw.csv', index=False)
    summary.to_csv('analysis/evaluation/eng_removal_bootstrapped_summary.csv', index=False)
    
    # --- VISUALIZATION ---
    plot_df = results_df.melt(id_vars=['Season', 'Rep'], var_name='Model', value_name='Accuracy')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x='Season', y='Accuracy', hue='Model', capsize=.1, errorbar='se')
    plt.title(f"Predictive Accuracy (N={n_boot} Bootstraps)\nTesting Impact of ENG & Blowout Removal")
    plt.ylim(0.5, 0.65)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analysis/evaluation/eng_removal_impact_bootstrapped.png', dpi=150)
    print("\nVisualization saved to analysis/evaluation/eng_removal_impact_bootstrapped.png")

if __name__ == "__main__":
    run_bootstrapped_eng_test(n_boot=100)
