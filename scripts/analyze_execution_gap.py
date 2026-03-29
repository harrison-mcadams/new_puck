
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.WARNING)

def analyze_execution_vs_accuracy(season='20232024'):
    # We need to run evaluation and get per-game or per-team Brier contributions
    evaluator_xg = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all', n_boot=0)
    evaluator_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all', n_boot=0)

    # Run a chronological split at 50%
    res_xg = evaluator_xg.run_evaluation(season, train_split=0.5, split_method='chronological')
    res_act = evaluator_act.run_evaluation(season, train_split=0.5, split_method='chronological')

    if not res_xg or not res_act:
        print("Evaluation failed.")
        return

    # Extract raw results (game-by-game p and y)
    df_xg = res_xg['Raw_Results']
    df_act = res_act['Raw_Results']
    
    # We need to map these back to teams to see which teams are "hard to predict" with xG
    # Since PredictiveEvaluator doesn't return team-id in Raw_Results, we need to recreate the loop
    df = DataUtils.load_season_data(season)
    sched_df = DataUtils.process_schedule(df)
    n_train = int(len(sched_df) * 0.5)
    test_sched = sched_df.iloc[n_train:].copy()

    # Get training abilities to see who is a "high finisher"
    train_df = df[df['game_id'].isin(sched_df.iloc[:n_train]['game_id'])]
    abilities_xg = evaluator_xg.summarizer.get_team_abilities(train_df, 'nested_xg', evaluator_xg.model_registry)
    abilities_act = evaluator_act.summarizer.get_team_abilities(train_df, 'actual', evaluator_act.model_registry)

    team_data = []
    for team in abilities_xg.keys():
        xg_f = abilities_xg[team]['for']
        act_f = abilities_act[team]['for']
        execution_gap = act_f - xg_f # Positive = Good finishing
        team_data.append({'Team': team, 'Execution_Gap': execution_gap})

    team_df = pd.DataFrame(team_data)
    
    # Now calculate Brier for each team in the test set
    test_results = []
    # Merge p_xg and p_act into test_sched
    test_sched['p_xg'] = df_xg['p'].values
    test_sched['p_act'] = df_act['p'].values
    test_sched['y'] = df_xg['y'].values

    for team in abilities_xg.keys():
        # Games where team is Home or Away
        mask = (test_sched['home_team'] == team) | (test_sched['away_team'] == team)
        team_games = test_sched[mask]
        
        if len(team_games) == 0: continue
        
        # Brier = (p - y)^2
        brier_xg = np.mean((team_games['p_xg'] - team_games['y'])**2)
        brier_act = np.mean((team_games['p_act'] - team_games['y'])**2)
        brier_diff = brier_xg - brier_act # Positive = Actual is better
        
        # update team_df
        idx = team_df[team_df['Team'] == team].index
        team_df.loc[idx, 'Brier_Diff'] = brier_diff
        team_df.loc[idx, 'n_test_games'] = len(team_games)

    print("\n--- Execution Gap vs Predictive Advantage (Actual vs xG) ---")
    print(team_df.sort_values('Execution_Gap', ascending=False).to_string(index=False))
    
    correlation = team_df['Execution_Gap'].corr(team_df['Brier_Diff'])
    print(f"\nCorrelation between Execution Gap and Actual-Goal Advantage: {correlation:.4f}")

if __name__ == "__main__":
    analyze_execution_vs_accuracy()
