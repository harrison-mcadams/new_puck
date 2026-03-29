
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.WARNING)

def test_standardization_fix(season='20232024'):
    evaluator = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)

    # 1. Base Run (Chronological 50/50)
    res_base = evaluator.run_evaluation(season, train_split=0.5, split_method='chronological')
    brier_base = res_base['Brier']
    acc_base = res_base['Accuracy']

    # 2. "Fixed" Run - Standardize abilities
    # To do this, we need to intercept the 'abilities' and scale them
    # I'll create a subclass or monkey-patch it for this experiment
    class FixedPredictiveEvaluator(PredictiveEvaluator):
        def run_evaluation(self, season, train_split=0.7, split_method='random', n_reps=1):
            # Same as base, but intercept abilities
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
            
            # --- THE FIX: Standardize xG Variance ---
            # Ratio found in analyze_std_fleet.py ~ 1.2
            STRETCH_RATIO = 1.2
            lg_avg = np.mean([v['for'] for v in abilities.values()])
            
            # Stretch 'for' and 'ag' away from league average
            for team in abilities:
                abilities[team]['for'] = lg_avg + (abilities[team]['for'] - lg_avg) * STRETCH_RATIO
                abilities[team]['ag'] = lg_avg + (abilities[team]['ag'] - lg_avg) * STRETCH_RATIO
                
                # Clip to positive
                abilities[team]['for'] = max(0.1, abilities[team]['for'])
                abilities[team]['ag'] = max(0.1, abilities[team]['ag'])
            # --- End Fix ---

            rep_results_list = []
            for _, row in test_sched_rep.iterrows():
                p_hw = self.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, lg_avg, self.outcome_type)
                actual = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0 if row['home_goals_final'] < row['away_goals_final'] else 0.5
                rep_results_list.append({'p': p_hw, 'y': actual})
                
            res_df = pd.DataFrame(rep_results_list)
            metrics = self.calculate_metrics(res_df, n_boot=0)
            return metrics

    evaluator_fixed = FixedPredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
    res_fixed = evaluator_fixed.run_evaluation(season, train_split=0.5, split_method='chronological')
    
    # 3. Actual Goals Run for Baseline
    evaluator_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0)
    res_act = evaluator_act.run_evaluation(season, train_split=0.5, split_method='chronological')

    print(f"\n--- Results for {season} (50/50 Chronological) ---")
    print(f"Base xG Brier:   {brier_base:.4f} (Acc: {acc_base:.4f})")
    print(f"Fixed xG Brier:  {res_fixed['Brier']:.4f} (Acc: {res_fixed['Accuracy']:.4f})")
    print(f"Actual Brier:    {res_act['Brier']:.4f} (Acc: {res_act['Accuracy']:.4f})")
    
    # Check if we beat Base xG and if we closed the gap with Actual
    if res_fixed['Brier'] < brier_base:
        improvement = (brier_base - res_fixed['Brier']) / brier_base * 100
        print(f"\nSUCCESS: Fixed model improved Brier by {improvement:.2f}%")
        if res_fixed['Brier'] < res_act['Brier']:
            print("EXTRA SUCCESS: Fixed xG now BEATS the Actual Goal model!")
    else:
        print("\nFAILURE: Standardizing variance did not improve Brier score.")

if __name__ == "__main__":
    test_standardization_fix('20232024')
    test_standardization_fix('20212022')
