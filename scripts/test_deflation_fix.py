
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.WARNING)

def test_deflation_and_variance_fix(season='20232024'):
    # Ratios for 2023-2024 from check_calibration_fleet.py:
    # xG Mean 3.1809, GF Mean 2.9813 (Ratio 1.067)
    # xG Std 0.2677, Act Std 0.3414 (Ratio 1.275)
    
    INFLATION_RATIO = 1.067
    STRETCH_RATIO = 1.2

    class FixedPredictiveEvaluator(PredictiveEvaluator):
        def run_evaluation(self, season, train_split=0.7, split_method='random', n_reps=1):
            df = DataUtils.load_season_data(season)
            sched_df = DataUtils.process_schedule(df)
            all_gids = np.array(sched_df['game_id'].values, dtype=int)
            total_games = len(sched_df)
            n_train = int(total_games * train_split)
            
            train_gids_rep = all_gids[:n_train]
            test_gids_rep = all_gids[n_train:]
            
            train_df_rep = df[df['game_id'].isin(train_gids_rep)].copy()
            test_sched_rep = sched_df[sched_df['game_id'].isin(test_gids_rep)].copy()
            
            abilities = self.summarizer.get_team_abilities(train_df_rep, self.model_name, self.model_registry)
            
            # --- THE FIXES ---
            lg_avg = np.mean([v['for'] for v in abilities.values()])
            
            # 1. Deflate to match GF Scale
            target_lg_avg = lg_avg / INFLATION_RATIO
            
            for team in abilities:
                # First, shift to target mean
                abilities[team]['for'] = abilities[team]['for'] / INFLATION_RATIO
                abilities[team]['ag'] = abilities[team]['ag'] / INFLATION_RATIO
                
                # Second, stretch variance
                abilities[team]['for'] = target_lg_avg + (abilities[team]['for'] - target_lg_avg) * STRETCH_RATIO
                abilities[team]['ag'] = target_lg_avg + (abilities[team]['ag'] - target_lg_avg) * STRETCH_RATIO
                
                # Clip to positive
                abilities[team]['for'] = max(0.1, abilities[team]['for'])
                abilities[team]['ag'] = max(0.1, abilities[team]['ag'])
            
            # Update lg_avg for internal Poisson math
            lg_avg = target_lg_avg
            # --- End Fixes ---

            rep_results_list = []
            for _, row in test_sched_rep.iterrows():
                p_hw = self.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, lg_avg, self.outcome_type)
                actual = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0 if row['home_goals_final'] < row['away_goals_final'] else 0.5
                rep_results_list.append({'p': p_hw, 'y': actual})
                
            res_df = pd.DataFrame(rep_results_list)
            metrics = self.calculate_metrics(res_df, n_boot=0)
            return metrics

    evaluator_base = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
    res_base = evaluator_base.run_evaluation(season, train_split=0.5, split_method='chronological')
    
    evaluator_fixed = FixedPredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
    res_fixed = evaluator_fixed.run_evaluation(season, train_split=0.5, split_method='chronological')
    
    evaluator_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0)
    res_act = evaluator_act.run_evaluation(season, train_split=0.5, split_method='chronological')

    print(f"\n--- Results for {season} (50/50 Chronological) ---")
    print(f"Base xG Brier:   {res_base['Brier']:.4f}")
    print(f"Fixed xG Brier:  {res_fixed['Brier']:.4f}")
    print(f"Actual Brier:    {res_act['Brier']:.4f}")

if __name__ == "__main__":
    test_deflation_and_variance_fix('20232024')
