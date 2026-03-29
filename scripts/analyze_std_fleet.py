
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils

logging.basicConfig(level=logging.WARNING)

def analyze_std_fleet():
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025']
    results = []

    for season in seasons:
        try:
            eval_xg = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all')
            eval_act = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all')

            df = DataUtils.load_season_data(season)
            abilities_xg = eval_xg.summarizer.get_team_abilities(df, 'nested_xg', eval_xg.model_registry)
            abilities_act = eval_act.summarizer.get_team_abilities(df, 'actual', eval_act.model_registry)

            xg_std = np.std([v['for'] for v in abilities_xg.values()])
            act_std = np.std([v['for'] for v in abilities_act.values()])
            
            results.append({
                'Season': season,
                'xG_Std': xg_std,
                'Act_Std': act_std,
                'Ratio': act_std / xg_std
            })
        except Exception as e:
            print(f"Error {season}: {e}")

    final_df = pd.DataFrame(results)
    print("\n--- Standard Deviation Mismatch Fleet-Wide ---")
    print(final_df.to_string(index=False))
    print(f"\nAverage Variance Mismatch Ratio: {final_df['Ratio'].mean():.4f}")

if __name__ == "__main__":
    analyze_std_fleet()
