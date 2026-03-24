import sys
import os
import pandas as pd
import numpy as np
from scripts import evaluate_predictive_power as eval_script

def test_scale_fix(season='20232024', model_name='xgboost_non_nested'):
    print(f"Testing Scale Fix for {model_name} in {season}...")
    
    # 1. Run Original
    evaluator = eval_script.PredictiveEvaluator(
        model_name=model_name, 
        metric_type='gd', 
        filter_type='all', 
        n_boot=0, 
        n_jobs=1
    )
    
    res_orig = evaluator.run_evaluation(season, train_split=0.7, split_method='random', n_reps=5)
    print(f"Original Brier: {res_orig['Brier']:.4f}, Accuracy: {res_orig['Accuracy']:.4f}")
    
    # 2. Run with Scale Fix (Manual override for now)
    # We'll patch the run_evaluation logic
    
    class PatchedEvaluator(eval_script.PredictiveEvaluator):
        def _run_single_rep_patched(self, r, df, sched_df, all_gids, n_train):
            rng = np.random.default_rng(seed=42 + r)
            train_gids_rep = rng.choice(all_gids, n_train, replace=False)
            test_gids_rep = np.array([g for g in all_gids if g not in train_gids_rep])
            train_df_rep = df[df['game_id'].isin(train_gids_rep)].copy()
            test_sched_rep = sched_df[sched_df['game_id'].isin(test_gids_rep)].copy()
            
            abilities = self.summarizer.get_team_abilities(train_df_rep, self.model_name, self.model_registry)
            
            # THE FIX: Use mean of abilities as league average
            league_avg_exp = np.mean([v['for'] for v in abilities.values()]) if abilities else 3.0
            
            rep_results_list = []
            for _, row in test_sched_rep.iterrows():
                p_hw = self.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, league_avg_exp, self.outcome_type)
                if row['home_goals_final'] > row['away_goals_final']: actual = 1.0
                elif row['home_goals_final'] < row['away_goals_final']: actual = 0.0
                else: actual = 0.5
                rep_results_list.append({'p': p_hw, 'y': actual})
            return eval_script.pd.DataFrame(rep_results_list)

    patcher = PatchedEvaluator(model_name=model_name, metric_type='gd', filter_type='all', n_boot=0, n_jobs=1)
    
    # Mocking run_evaluation behavior
    df = eval_script.DataUtils.load_season_data(season)
    sched_df = eval_script.DataUtils.process_schedule(df)
    all_gids = np.array(sched_df['game_id'].values, dtype=int)
    n_train = int(len(sched_df) * 0.7)
    
    results = [patcher._run_single_rep_patched(r, df, sched_df, all_gids, n_train) for r in range(5)]
    combined = pd.concat(results)
    
    # Calculate metrics
    y_true = combined['y'].values
    y_pred = combined['p'].values
    brier = np.mean((y_true - y_pred)**2)
    acc = np.mean((y_pred > 0.5) == (y_true > 0.5))
    
    print(f"Patched  Brier: {brier:.4f}, Accuracy: {acc:.4f}")
    
    # Compare to Actual
    actual_eval = eval_script.PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0, n_jobs=1)
    res_actual = actual_eval.run_evaluation(season, train_split=0.7, split_method='random', n_reps=5)
    print(f"Actual   Brier: {res_actual['Brier']:.4f}, Accuracy: {res_actual['Accuracy']:.4f}")

if __name__ == "__main__":
    test_scale_fix('20232024', 'xgboost_non_nested')
    test_scale_fix('20232024', 'nested_xg')
