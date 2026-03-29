import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import features as feature_util
from puck import fit_xgboost_nested, fit_xgboost_non_nested, mixed_effects
from scripts.evaluate_predictive_power import PredictiveEvaluator, DataUtils, ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpecializedModelRegistry(ModelRegistry):
    """Overrides ModelRegistry to support specific feature sets for the comparison study."""
    def __init__(self, feature_mode='all_inclusive'):
        super().__init__()
        self.feature_mode = feature_mode
        self.standard_features = feature_util.get_features('standard')
        self.all_features = feature_util.get_features('all_inclusive')

    def _train_local_model(self, model_name, train_df):
        if train_df is None:
            return None
        
        # Select features based on mode
        # 'with_time' means all_inclusive (includes season)
        # 'no_time' means standard (excludes season)
        if self.feature_mode == 'with_time':
            feature_list = self.all_features
        else:
            feature_list = self.standard_features
            
        logger.debug(f"Training local model {model_name} with features: {len(feature_list)}")
        
        if model_name in ['xgboost_nested', 'nested_xg']:
            model = fit_xgboost_nested.XGBNestedXGClassifier(features=feature_list)
            model.fit(train_df)
        elif model_name in ['xgboost_non_nested', 'non_nested_xg']:
            model = fit_xgboost_non_nested.XGBNonNestedXGClassifier(features=feature_list)
            model.fit(train_df)
        else:
            # Fallback to base registry for other models if any
            return super()._train_local_model(model_name, train_df)
        
        return model

def run_accuracy_study(n_reps=100, seed=42):
    """Executes the accuracy comparison study as requested."""
    out_dir = Path("analysis/accuracy_violins")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_seasons = DataUtils.get_available_seasons()
    modern_seasons = [s for s in all_seasons if int(s) >= 20202021]
    logger.info(f"Running study across seasons: {modern_seasons}")
    
    # Study 1: Filter Comparison (xgboost_nested)
    # Filters: all, 5v5, 5v5_close, extrapolated_per60 vs actual
    filter_results = []
    filters = ['all', '5v5', '5v5_close', 'extrapolated_per60']
    
    logger.info("\n--- PHASE 1: Filter Comparison (xgboost_nested) ---")
    for f in filters + ['actual']:
        # For actual goals, we use 'all' filter as baseline unless specified
        # but 'actual' in PredictiveEvaluator handles this.
        model_name = 'xgboost_nested' if f != 'actual' else 'actual'
        eval_f = f if f != 'actual' else 'all'
        
        evaluator = PredictiveEvaluator(model_name, 'gd', eval_f, n_boot=0, seed=seed, n_jobs=-1, no_dashboards=True)
        # Use our Specialized Registry (defaulting to standard for filters)
        evaluator.model_registry = SpecializedModelRegistry(feature_mode='no_time')
        
        season_accs = []
        for season in modern_seasons:
            res = evaluator.run_evaluation(season, n_reps=n_reps)
            if res:
                season_accs.extend(res.get('Accuracy_dist', []))
        
        for acc in season_accs:
            filter_results.append({
                'Filter': f,
                'Accuracy': acc * 100,
                'Comparison': 'Filter Impact'
            })

    # Plot Study 1
    if filter_results:
        plt.figure(figsize=(10, 6))
        df1 = pd.DataFrame(filter_results)
        sns.violinplot(data=df1, x='Filter', y='Accuracy', inner='quartile', palette='muted')
        plt.title('Accuracy Comparison: Impact of Game-State Filters (Nested XGBoost)', fontsize=14, fontweight='bold')
        plt.ylabel('Game Outcome Accuracy (%)')
        plt.axhline(50, color='red', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'accuracy_by_filter.png', dpi=300)
        plt.close()
        logger.info(f"Filter plot saved to {out_dir / 'accuracy_by_filter.png'}")

    # Study 2: Booster Modifier Comparison (Nested vs Non-Nested +/- Season)
    # Models: Nested, Non-Nested, Nested+Time, Non-Nested+Time vs Actual
    modifier_results = []
    variants = [
        ('Nested', 'xgboost_nested', 'no_time'),
        ('Non-Nested', 'xgboost_non_nested', 'no_time'),
        ('Nested + Time', 'xgboost_nested', 'with_time'),
        ('Non-Nested + Time', 'xgboost_non_nested', 'with_time'),
        ('Actual Goals', 'actual', 'no_time')
    ]
    
    logger.info("\n--- PHASE 2: Booster Modifier Comparison ---")
    for label, model_name, f_mode in variants:
        evaluator = PredictiveEvaluator(model_name, 'gd', 'all', n_boot=0, seed=seed, n_jobs=-1, no_dashboards=True)
        evaluator.model_registry = SpecializedModelRegistry(feature_mode=f_mode)
        
        season_accs = []
        for season in modern_seasons:
            # We use 'local_' prefix to trigger _train_local_model in evaluator logic
            # but wait, evaluator.run_evaluation uses self.model_name
            # If self.model_name starts with local_, it works.
            evaluator.model_name = f"local_{model_name}" if model_name != 'actual' else 'actual'
            
            res = evaluator.run_evaluation(season, n_reps=n_reps)
            if res:
                season_accs.extend(res.get('Accuracy_dist', []))
        
        for acc in season_accs:
            modifier_results.append({
                'Model': label,
                'Accuracy': acc * 100,
                'Comparison': 'Modifier Impact'
            })

    # Plot Study 2
    if modifier_results:
        plt.figure(figsize=(10, 6))
        df2 = pd.DataFrame(modifier_results)
        sns.violinplot(data=df2, x='Model', y='Accuracy', inner='quartile', palette='viridis')
        plt.title('Accuracy Comparison: Booster Modifiers & Seasonal Context', fontsize=14, fontweight='bold')
        plt.ylabel('Game Outcome Accuracy (%)')
        plt.axhline(50, color='red', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'accuracy_by_modifier.png', dpi=300)
        plt.close()
        logger.info(f"Modifier plot saved to {out_dir / 'accuracy_by_modifier.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=100)
    args = parser.parse_args()
    
    run_accuracy_study(n_reps=args.reps)
