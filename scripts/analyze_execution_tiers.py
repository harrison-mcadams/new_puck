
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

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_bootstrapped_tier_analysis(n_boot=100, seed=42):
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025']
    rng = np.random.default_rng(seed)
    
    all_game_results = []
    
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
                
                # 1. Setup Models & Extract Abilities
                eval_xg = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
                eval_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0)
                
                abs_xg = eval_xg.summarizer.get_team_abilities(train_df, 'nested_xg', eval_xg.model_registry)
                abs_act = eval_act.summarizer.get_team_abilities(train_df, 'actual', eval_act.model_registry)
                
                # 2. Calculate Execution Index
                team_tiers = {}
                execution_stats = []
                for team in abs_xg:
                    idx = (abs_act[team]['for'] - abs_xg[team]['for']) + (abs_xg[team]['ag'] - abs_act[team]['ag'])
                    execution_stats.append({'team': team, 'idx': idx})
                
                exec_df = pd.DataFrame(execution_stats)
                t_hi = exec_df['idx'].quantile(0.85)
                t_lo = exec_df['idx'].quantile(0.15)
                
                for _, row in exec_df.iterrows():
                    tier = 'Elite' if row['idx'] >= t_hi else 'Poor' if row['idx'] <= t_lo else 'Average'
                    team_tiers[row['team']] = tier
                
                # 3. Predict Winners
                lg_avg_xg = np.mean([v['for'] for v in abs_xg.values()])
                lg_avg_act = np.mean([v['for'] for v in abs_act.values()])
                
                for _, row in test_sched.iterrows():
                    h, a = row['home_team'], row['away_team']
                    if h not in team_tiers or a not in team_tiers: continue
                    
                    p_xg = eval_xg.matchup_engine.predict_winner_prob(h, a, abs_xg, lg_avg_xg, 'winner')
                    p_act = eval_act.matchup_engine.predict_winner_prob(h, a, abs_act, lg_avg_act, 'winner')
                    y = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0
                    
                    h_tier, a_tier = team_tiers[h], team_tiers[a]
                    matchup_type = " vs ".join(sorted([h_tier, a_tier]))
                    
                    all_game_results.append({
                        'Matchup_Type': matchup_type,
                        'Correct_XG': 1 if (p_xg >= 0.5 and y == 1.0) or (p_xg < 0.5 and y == 0.0) else 0,
                        'Correct_Act': 1 if (p_act >= 0.5 and y == 1.0) or (p_act < 0.5 and y == 0.0) else 0
                    })
        except Exception as e:
            print(f"Error in {season}: {e}")

    results_df = pd.DataFrame(all_game_results)
    summary = results_df.groupby('Matchup_Type').agg(['mean', 'count']).reset_index()
    summary.columns = ['Matchup_Type', 'Acc_XG', 'N_Games', 'Acc_Act', 'N_Games_Duplicate']
    summary['Improvement'] = summary['Acc_Act'] - summary['Acc_XG']
    
    print("\n--- Bootstrapped Tier Matchup Analysis ---")
    print(summary[['Matchup_Type', 'N_Games', 'Acc_XG', 'Acc_Act', 'Improvement']].sort_values('Improvement', ascending=False).to_string(index=False))
    
    os.makedirs('analysis/evaluation', exist_ok=True)
    results_df.to_csv('analysis/evaluation/execution_tier_bootstrapped_raw.csv', index=False)
    summary.to_csv('analysis/evaluation/execution_tier_bootstrapped_summary.csv', index=False)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 7))
    sns.barplot(data=summary, x='Improvement', y='Matchup_Type', hue='Matchup_Type', palette='vlag', legend=False)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Predictive Accuracy Advantage: Actual minus xG\n(N={n_boot} Bootstraps across Modern Era 2020-2025)")
    plt.xlabel("Accuracy Advantage (Actual % - xG %)")
    plt.tight_layout()
    plt.savefig('analysis/evaluation/execution_tier_accuracy_bootstrapped.png', dpi=150)
    print("\nVisualization saved to analysis/evaluation/execution_tier_accuracy_bootstrapped.png")

if __name__ == "__main__":
    run_bootstrapped_tier_analysis(n_boot=100)
