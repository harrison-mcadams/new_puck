
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PredictiveEvaluator, ModelRegistry, TeamAbilitySummarizer, DataUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_variance(season='20232024'):
    evaluator_xg = PredictiveEvaluator(model_name='nested_xg', metric_type='gd', filter_type='all')
    evaluator_actual = PredictiveEvaluator(model_name='actual', metric_type='gd', filter_type='all')

    df = DataUtils.load_season_data(season)
    # Use a 1.0 split to get full season abilities for comparison
    train_df = df.copy()

    # Get xG abilities
    abilities_xg = evaluator_xg.summarizer.get_team_abilities(train_df, 'nested_xg', evaluator_xg.model_registry)
    # Get Actual abilities
    abilities_actual = evaluator_actual.summarizer.get_team_abilities(train_df, 'actual', evaluator_actual.model_registry)

    xg_fors = [v['for'] for v in abilities_xg.values()]
    actual_fors = [v['for'] for v in abilities_actual.values()]

    logger.info(f"Season: {season}")
    logger.info(f"xG For - Mean: {np.mean(xg_fors):.4f}, Std: {np.std(xg_fors):.4f}, Var: {np.var(xg_fors):.4f}")
    logger.info(f"Actual For - Mean: {np.mean(actual_fors):.4f}, Std: {np.std(actual_fors):.4f}, Var: {np.var(actual_fors):.4f}")

    # Compare distributions
    diff_std = np.std(actual_fors) / np.std(xg_fors) if np.std(xg_fors) > 0 else 0
    logger.info(f"Actual Std is {diff_std:.2f}x the xG Std")

if __name__ == "__main__":
    analyze_variance()
