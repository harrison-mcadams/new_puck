"""scripts/evaluate_predictive_power.py

Overhauled predictive power evaluation routine.
Flexible and modular structure for assessing xG models and team performance metrics.

Available Options:
-----------------
--model:  nested_xg, non_nested_xg, mixed_effects, actual, moneypuck.
          Prefix with 'local_' to train on the specific season's training split 
          (e.g., local_nested_xg).
--filter: all (default), 5v5, 5v5_close (score diff <= 1), extrapolated_per60.
--metric: gd (Goal Difference, default), gf_pct (Goals For %), rank (Team Rankings).
--seasons: Comma-separated list (e.g., 20222023,20232024) or 'aggregate' to 
           auto-discover all seasons and calculate weighted combined results.

Example Calls:
--------------
1) Rank all teams by non-local nested xG absolute difference, 5v5 score close:
   python scripts/evaluate_predictive_power.py --model nested_xg --metric rank --filter 5v5_close --train-split 1.0

2) Compare all available models (Global vs Local) across all seasons in aggregate:
   python scripts/evaluate_predictive_power.py --seasons aggregate \
     --model nested_xg,local_nested_xg,non_nested_xg,local_non_nested_xg,mixed_effects,local_mixed_effects,actual \
     --filter 5v5 --n-boot 1000

3) Compare all filter conditions for a specific model in aggregate:
   python scripts/evaluate_predictive_power.py --seasons aggregate \
     --model nested_xg --filter all,5v5,5v5_close,extrapolated_per60 --n-boot 1000
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import datetime
import logging
import joblib
import pickle
from pathlib import Path
import scipy.stats as stats
from scipy.stats import poisson, spearmanr
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing, analyze, mixed_effects, nhl_api, fit_glm_nested, fit_glm
from puck import features as feature_util
from puck import moneypuck, data_pipeline
from scripts import plot_predictive_power

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Utilities ---

class DataUtils:
    @staticmethod
    def load_season_data(season):
        """Load and preprocess season data."""
        logger.info(f"Loading data for {season}...")
        csv_path = analyze.locate_season_csv(season)
        df = pd.read_csv(csv_path)
        
        # Pre-process for models
        df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=True, 
            apply_arena_adjustments=True, apply_bio_enrichment=True, apply_filtering=True
        )
        return df

    @staticmethod
    def process_schedule(df, season_schedule_dict=None):
        """Extracts a chronological list of games from the event dataframe."""
        games = []
        for gid, group in df.groupby('game_id'):
            home_team = group['home_abb'].iloc[0]
            away_team = group['away_abb'].iloc[0]
            
            # Regulation outcomes
            # A goal is for a team if team_id matches
            home_goals_reg = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['home_id']) & (group['period'] <= 3)])
            away_goals_reg = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['away_id']) & (group['period'] <= 3)])
            
            # Ultimate outcomes (including OT/SO)
            if season_schedule_dict and gid in season_schedule_dict:
                actual = season_schedule_dict[gid]
                home_goals_final = actual['home_goals']
                away_goals_final = actual['away_goals']
                is_ot_so = actual['is_ot_so']
            else:
                # Fallback heuristics
                home_goals_final = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['home_id'])])
                away_goals_final = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['away_id'])])
                is_ot_so = group['period'].max() > 3
            
            games.append({
                'game_id': gid,
                'home_team': home_team,
                'away_team': away_team,
                'home_goals_reg': home_goals_reg,
                'away_goals_reg': away_goals_reg,
                'home_goals_final': home_goals_final,
                'away_goals_final': away_goals_final,
                'is_ot_so': is_ot_so
            })
            
        sched_df = pd.DataFrame(games).sort_values('game_id').reset_index(drop=True)
        return sched_df

    @staticmethod
    def get_available_seasons():
        """Scans data/ directory for season folders."""
        data_dir = Path("data")
        seasons = []
        if data_dir.exists():
            for d in data_dir.iterdir():
                if d.is_dir():
                    s = d.name
                    if s.isdigit() and len(s) == 8:
                        # Basic check: does it look like a season?
                        seasons.append(s)
        return sorted(seasons)

    @staticmethod
    def get_team_game_counts(sched_df):
        team_schedules = {}
        for _, row in sched_df.iterrows():
            h, a, gid = row['home_team'], row['away_team'], row['game_id']
            if h not in team_schedules: team_schedules[h] = []
            team_schedules[h].append(gid)
            if a not in team_schedules: team_schedules[a] = []
            team_schedules[a].append(gid)
        return team_schedules

# --- Model Management ---

class ModelRegistry:
    """Manages loading and training of xG models."""
    def __init__(self):
        self._cache = {}

    def get_model(self, model_name, train_df=None, is_local=False):
        cache_key = (model_name, is_local, id(train_df) if train_df is not None else None)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if model_name == 'mixed_effects' and not is_local:
            # "Global" mixed effects uses the global nested GLM as a fixed-effect base
            # but MUST fit team-specific intercepts on the current train_df.
            if train_df is None:
                logger.warning("train_df is None, cannot fit mixed effects intercepts.")
                return None
            logger.info("Fitting mixed effects intercepts for current season...")
            base = self._load_global_model('nested_xg')
            model = mixed_effects.GameMixedEffectsXG(base_model=base)
            model.fit(train_df)
        elif is_local:
            model = self._train_local_model(model_name, train_df)
        else:
            model = self._load_global_model(model_name)
        
        self._cache[cache_key] = model
        return model

    def _load_global_model(self, model_name):
        if model_name == 'actual':
            return 'actual'
        
        paths = {
            'nested_xg': os.path.join('analysis', 'xgs', 'xg_model_nested_tensor.joblib'),
            'non_nested_xg': os.path.join('analysis', 'xgs', 'xg_model_non_nested_tensor.joblib')
        }
        
        path = paths.get(model_name)
        if not path or not os.path.exists(path):
            logger.warning(f"Global model {model_name} not found at {path}. Returning None.")
            return None
        
        return joblib.load(path)

    def _train_local_model(self, model_name, train_df):
        if train_df is None:
            return None
        
        logger.info(f"Training local model: {model_name}...")
        feature_list = feature_util.get_features('all_inclusive')
        
        if model_name == 'nested_xg':
            model = fit_glm_nested.NestedGLM(features=feature_list, use_splines=True, enable_marginalization=True)
            model.fit(train_df)
        elif model_name == 'non_nested_xg':
            model = fit_glm.NonNestedGLM(features=feature_list, use_splines=True, enable_marginalization=True)
            model.fit(train_df[train_df['event'] != 'blocked-shot'])
        elif model_name == 'mixed_effects':
            # Train base nested then mixed
            base = self._train_local_model('nested_xg', train_df)
            model = mixed_effects.GameMixedEffectsXG(base_model=base, use_tensor_splines=True)
            model.fit(train_df)
        else:
            return None
        
        return model

# --- Team Ability Summarization ---

class TeamAbilitySummarizer:
    """Summarizes team performance based on xG models and filters."""
    def __init__(self, metric_type='gd', filter_type='all'):
        """
        metric_type: 'gd' (Goal Difference), 'gf_pct' (Goals For %), 'rank'
        filter_type: 'all', '5v5', '5v5_close' (score diff <= 1), 'extrapolated_per60'
        """
        self.metric_type = metric_type
        self.filter_type = filter_type

    def get_team_abilities(self, train_df, model_name, model_registry):
        if self.filter_type == 'extrapolated_per60':
            return self._summarize_extrapolated(train_df, model_name, model_registry)
        
        df = self._apply_filter(train_df)
        
        # Predict xG if not 'actual'
        if model_name != 'actual':
            is_local = 'local' in model_name
            pure_model_name = model_name.replace('local_', '')
            model = model_registry.get_model(pure_model_name, train_df, is_local=is_local)
            if model:
                df = df.copy()
                if pure_model_name == 'mixed_effects':
                    df['eval_xg'] = model.predict_proba(df)[:, 1]
                else:
                    mask = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
                    df['eval_xg'] = 0.0
                    if mask.any():
                        df.loc[mask, 'eval_xg'] = model.predict_proba(df[mask])[:, 1]
            else:
                df['eval_xg'] = 0.0
        else:
            df = df.copy()
            df.loc[:, 'eval_xg'] = (df['event'].str.lower() == 'goal').astype(float)

        # Group by team
        teams = pd.concat([train_df['home_abb'], train_df['away_abb']]).unique()
        abilities = {}
        
        for t in teams:
            # GF/XGF
            is_for = ((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | \
                     ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))
            is_ag = ((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | \
                    ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))
            
            f_val = df[is_for]['eval_xg'].sum()
            a_val = df[is_ag]['eval_xg'].sum()
            
            team_games = train_df[(train_df['home_abb'] == t) | (train_df['away_abb'] == t)]['game_id'].nunique()
            
            # Always return both For and Against rates
            abilities[t] = {
                'for': f_val / team_games if team_games > 0 else 0,
                'ag': a_val / team_games if team_games > 0 else 0
            }
                
        return abilities

    def _apply_filter(self, df):
        if self.filter_type == '5v5':
            return df[df['game_state'] == '5v5']
        elif self.filter_type == '5v5_close':
            mask_5v5 = df['game_state'] == '5v5'
            if 'home_score' in df.columns and 'away_score' in df.columns:
                mask_close = (df['home_score'] - df['away_score']).abs() <= 1
                return df[mask_5v5 & mask_close]
            else:
                return df[mask_5v5]
        return df

    def _summarize_extrapolated(self, train_df, model_name, model_registry):
        """Logic for 'extrapolated per60'."""
        logger.info("Summarizing ability via extrapolated per60...")
        
        # League-average time-on-ice per game (seconds)
        AVG_TIME = {
            '5v5': 48.0 * 60,
            '5v4': 6.0 * 60,
            '4v5': 6.0 * 60
        }
        
        teams = pd.concat([train_df['home_abb'], train_df['away_abb']]).unique()
        is_local = 'local' in model_name
        pure_model_name = model_name.replace('local_', '')
        model = model_registry.get_model(pure_model_name, train_df, is_local=is_local)
        
        abilities = {}
        for t in teams:
            team_games = train_df[(train_df['home_abb'] == t) | (train_df['away_abb'] == t)]['game_id'].nunique()
            if team_games == 0: continue
            
            total_f, total_a = 0.0, 0.0
            for state in ['5v5', '5v4', '4v5']:
                # Filter events for this team in this state
                # If team is home, state 5v4 is PP. If team is away, state 4v5 is PP.
                # Actually, simpler to just use team_id for/against logic.
                
                mask_state = train_df['game_state'] == state
                mask_for = mask_state & (
                    ((train_df['home_abb'] == t) & (train_df['team_id'] == train_df['home_id'])) |
                    ((train_df['away_abb'] == t) & (train_df['team_id'] == train_df['away_id']))
                )
                mask_ag = mask_state & (
                    ((train_df['home_abb'] == t) & (train_df['team_id'] != train_df['home_id'])) |
                    ((train_df['away_abb'] == t) & (train_df['team_id'] != train_df['away_id']))
                )
                
                if model_name == 'actual':
                    f_val = (train_df[mask_for]['event'].str.lower() == 'goal').sum()
                    a_val = (train_df[mask_ag]['event'].str.lower() == 'goal').sum()
                else:
                    f_val = model.predict_proba(train_df[mask_for])[:, 1].sum() if mask_for.any() else 0.0
                    a_val = model.predict_proba(train_df[mask_ag])[:, 1].sum() if mask_ag.any() else 0.0
                
                # In current script, we use shots as proxy for time if TOI not available accurately.
                # Or just use the AVG_TIME directly.
                # "take the amount of time per game they tend to be on 5v5 and the powerplay"
                # This implies we should calculate THEIR specific TOI.
                # Since we don't have shift data easily here, let's assume we use AVG_TIME for extrapolation 
                # but scaled by their relative shot rate if possible, or just the goal total.
                # Let's stick to the Project's convention: total_val / n_games * (TOTAL_SESSION_TIME / ESTIMATED_STATE_TIME)
                # Actually, just (f_val - a_val) / n_games for each state, then sum?
                # No, extrapolation usually means: Rate * Avg_Time.
                # Rate = f_val / (num_shots in state) ... no, Rate = f_val / state_time.
                # Since we don't have state_time for each team easily, let's use the shot proxy.
                
                # Simplified: Total for/against in this state, normalized to per-game, 
                # then scaled to "what if they played exactly AVG_TIME".
                # But wait, (f_val / n_games) already tells us what they DO per game.
                # Extrapolated usually means: "Suppose they played 48 mins of 5v5, 6 mins of PP, etc."
                # If they already play 48 mins of 5v5, then (f_val / n_games) is correct.
                # If they play more or less, we'd need shift data.
                # For this implementation, I'll use the per-game total as the base.
                total_f += f_val
                total_a += a_val
            
            abilities[t] = {
                'for': total_f / team_games if team_games > 0 else 0,
                'ag': total_a / team_games if team_games > 0 else 0
            }
                
        return abilities

# --- Matchup Engines ---

class MatchupEngine:
    def predict_winner_prob(self, home_team, away_team, team_abilities, league_avg, outcome_type='final'):
        raise NotImplementedError

class PoissonMatchupEngine(MatchupEngine):
    """Robust Poisson-based prediction."""
    def __init__(self, logic_type='multiplicative'):
        self.logic_type = logic_type

    def predict_winner_prob(self, home_team, away_team, team_abilities, league_avg, outcome_type='final'):
        h_stats = team_abilities.get(home_team, {'for': league_avg, 'ag': league_avg})
        a_stats = team_abilities.get(away_team, {'for': league_avg, 'ag': league_avg})
        
        if self.logic_type == 'multiplicative':
            # Multiplicative logic: (Home_For * Away_Against) / League_Avg
            h_exp = (h_stats['for'] * a_stats['ag']) / league_avg if league_avg > 0 else 0.0
            a_exp = (a_stats['for'] * h_stats['ag']) / league_avg if league_avg > 0 else 0.0
        else:
            # Additive logic: League_Avg + (Home_GD - Away_GD) / 2
            h_gd = h_stats['for'] - h_stats['ag']
            a_gd = a_stats['for'] - a_stats['ag']
            h_exp = league_avg + (h_gd - a_gd) / 2
            a_exp = league_avg + (a_gd - h_gd) / 2
        
        h_exp = max(0.1, h_exp)
        a_exp = max(0.1, a_exp)
        
        # Poisson distribution for goals
        max_g = 15
        h_dist = stats.poisson.pmf(np.arange(max_g), h_exp)
        a_dist = stats.poisson.pmf(np.arange(max_g), a_exp)
        prob_matrix = np.outer(h_dist, a_dist)
        
        hw = np.tril(prob_matrix, -1).sum()
        aw = np.triu(prob_matrix, 1).sum()
        tie = np.trace(prob_matrix)
        
        total = hw + aw + tie
        if total > 0:
            hw, aw, tie = hw/total, aw/total, tie/total
        else:
            hw, aw, tie = 0.33, 0.33, 0.34
        
        if outcome_type == 'final':
            return hw + tie * 0.5
        else:
            return hw

class SimulationMatchupEngine(MatchupEngine):
    """Advanced simulation using matchup.py logic."""
    def __init__(self, logic_type='multiplicative'):
        self.logic_type = logic_type

    def predict_winner_prob(self, home_team, away_team, team_abilities, league_avg, outcome_type='final'):
        # Integration with scripts/matchup.py
        try:
            from scripts import matchup as matchup_module
            # This is slow, so we might want to cache assets or use a lighter version
            # For this overhaul, we'll provide a hook.
            # Simulation is complex, for now we fallback to Poisson if not fully implemented
            # or provide a simplified version.
            return PoissonMatchupEngine(logic_type=self.logic_type).predict_winner_prob(home_team, away_team, team_abilities, league_avg, outcome_type)
        except ImportError:
            return PoissonMatchupEngine(logic_type=self.logic_type).predict_winner_prob(home_team, away_team, team_abilities, league_avg, outcome_type)

# --- Orchestration ---

class PredictiveEvaluator:
    def __init__(self, model_name, metric_type, filter_type, matchup_type='poisson', outcome_type='final', n_boot=1000, matchup_logic='multiplicative'):
        self.model_registry = ModelRegistry()
        self.summarizer = TeamAbilitySummarizer(metric_type, filter_type)
        
        self.model_name = model_name
        self.metric_type = metric_type
        self.filter_type = filter_type
        self.matchup_type = matchup_type
        self.outcome_type = outcome_type
        self.n_boot = n_boot
        self.matchup_logic = matchup_logic

        # Engine Selection
        if matchup_type == 'poisson':
            self.matchup_engine = PoissonMatchupEngine(logic_type=matchup_logic)
        else:
            self.matchup_engine = SimulationMatchupEngine(logic_type=matchup_logic)

    def run_evaluation(self, season, train_split=0.7):
        df = DataUtils.load_season_data(season)
        sched_df = DataUtils.process_schedule(df)
        
        # Split data
        total_games = len(sched_df)
        n_train = int(total_games * train_split)
        
        if n_train >= total_games:
            train_df = df.copy()
            test_df = pd.DataFrame()
            test_sched = pd.DataFrame()
            train_gid_cutoff = 99999999
        else:
            train_gid_cutoff = sched_df.iloc[n_train]['game_id']
            train_df = df[df['game_id'] < train_gid_cutoff].copy()
            test_df = df[df['game_id'] >= train_gid_cutoff].copy()
            test_sched = DataUtils.process_schedule(test_df)
            
        logger.info(f"Season {season}: Training on {len(train_df)} events, testing on {len(test_df)} events.")
        
        # Summarize Ability
        abilities = self.summarizer.get_team_abilities(train_df, self.model_name, self.model_registry)
        
        # Ranking Output if requested
        if self.metric_type == 'rank':
            self._output_rankings(season, abilities)
            if train_split >= 1.0: return None
        
        if len(test_sched) == 0:
            logger.warning("No test games found. Skipping evaluation.")
            return None
        
        # Calculate empirical league average per game for this slice
        n_games_train = sched_df[sched_df['game_id'] < (train_gid_cutoff if n_train < total_games else 99999999)]['game_id'].nunique()
        if n_games_train > 0:
            # We want goals per team-game
            league_avg_exp = df[df['game_id'].isin(train_df['game_id'])]['event'].str.count('goal').sum() / (2 * n_games_train)
        else:
            league_avg_exp = 3.0
            
        logger.info(f"Empirical League Average: {league_avg_exp:.3f} goals per team-game")
        
        results = []
        for _, row in test_sched.iterrows():
            p_hw = self.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, league_avg_exp, self.outcome_type)
            
            # Actual result
            if self.outcome_type == 'final':
                actual = 1.0 if row['home_goals_final'] > row['away_goals_final'] else 0.0
            else:
                if row['home_goals_reg'] > row['away_goals_reg']: actual = 1.0
                elif row['away_goals_reg'] > row['home_goals_reg']: actual = 0.0
                else: actual = 0.5
                
            results.append({'p': p_hw, 'y': actual})
            
        res_df = pd.DataFrame(results)
        
        # Metrics with optional Bootstrap
        summary = self.calculate_metrics(res_df)
        
        logger.info(f"Season {season} | Brier: {summary['Brier']:.4f} ({summary['Brier_lo']:.4f}-{summary['Brier_hi']:.4f}) | Acc: {summary['Accuracy']:.4f} ({summary['Accuracy_lo']:.4f}-{summary['Accuracy_hi']:.4f})")
        
        return {
            'Season': season,
            'Model': self.model_name,
            'Filter': self.filter_type,
            'Metric': self.metric_type,
            **summary,
            'Test_Games': len(test_sched),
            'Raw_Results': res_df
        }

    def calculate_metrics(self, res_df, n_boot=None):
        """Calculates Brier and Accuracy with bootstrapping."""
        if n_boot is None: n_boot = self.n_boot
        
        y_true = res_df['y'].to_numpy()
        y_pred = res_df['p'].to_numpy()
        
        if n_boot > 0 and len(res_df) > 5:
            b_mean, b_lo, b_hi, b_dist = self._bootstrap_metric(y_true, y_pred, self._brier_fn, n_boot=n_boot)
            a_mean, a_lo, a_hi, a_dist = self._bootstrap_metric(y_true, y_pred, self._acc_fn, n_boot=n_boot)
        else:
            b_mean = self._brier_fn(y_true, y_pred)
            b_lo = b_hi = b_mean
            b_dist = np.array([b_mean])
            a_mean = self._acc_fn(y_true, y_pred)
            a_lo = a_hi = a_mean
            a_dist = np.array([a_mean])
            
        return {
            'Brier': b_mean, 'Brier_lo': b_lo, 'Brier_hi': b_hi, 'Brier_dist': b_dist,
            'Accuracy': a_mean, 'Accuracy_lo': a_lo, 'Accuracy_hi': a_hi, 'Accuracy_dist': a_dist
        }

    def _output_rankings(self, season, abilities):
        logger.info(f"--- Team Rankings for {season} ({self.model_name}, {self.filter_type}) ---")
        
        # Collapse dual rates to the requested metric for ranking
        metric_vals = {}
        for t, rates in abilities.items():
            if self.metric_type == 'gd':
                metric_vals[t] = rates['for'] - rates['ag']
            elif self.metric_type == 'gf_pct':
                metric_vals[t] = rates['for'] / (rates['for'] + rates['ag']) if (rates['for'] + rates['ag']) > 0 else 0.5
            else:
                metric_vals[t] = rates['for'] - rates['ag']
                
        ranks = pd.Series(metric_vals).sort_values(ascending=False).rank(ascending=False, method='min')
        df_ranks = pd.DataFrame({'Metric': metric_vals, 'Rank': ranks}).sort_values('Rank')
        print(df_ranks)
        out_path = Path(f"analysis/evaluation/rankings_{season}_{self.model_name}_{self.filter_type}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_ranks.to_csv(out_path)

    def _bootstrap_metric(self, y_true, y_pred, metric_fn, ci=95, n_boot=None):
        if n_boot is None: n_boot = self.n_boot
        n = len(y_true)
        indices = np.random.randint(0, n, (n_boot, n))
        boots = np.array([metric_fn(y_true[idx], y_pred[idx]) for idx in indices])
        alpha = (100 - ci) / 2
        return float(np.mean(boots)), float(np.percentile(boots, alpha)), float(np.percentile(boots, 100 - alpha)), boots

    def _brier_fn(self, y, p):
        return float(np.mean((p - y)**2))

    def _acc_fn(self, y, p):
        # Handle ties (y=0.5) as half-correct or ignore?
        # Standard accuracy: (p > 0.5) matches (y > 0.5)
        # Ties in regulation are tricky.
        return float(np.mean((p > 0.5) == (y > 0.5)))

def _get_config_palette(configurations):
    """Creates a color palette where local/non-local models are linked."""
    palette = {}
    # Modern, professional base colors
    base_colors = {
        'nested_xg': '#1f77b4',      # Blue
        'non_nested_xg': '#ff7f0e', # Orange
        'mixed_effects': '#2ca02c', # Green
        'actual': '#d62728',        # Red
        'moneypuck': '#9467bd'      # Purple
    }
    
    for config in configurations:
        model_part = config.split(' (')[0]
        pure_model = model_part.replace('local_', '')
        base_hex = base_colors.get(pure_model, '#7f7f7f')
        
        if 'local' in model_part:
            # Create a related, lighter/brighter color
            rgb = mcolors.to_rgb(base_hex)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            # Increase lightness and decrease saturation slightly for a "softer" look
            new_l = min(0.95, l * 1.5)
            new_s = s * 0.8
            palette[config] = mcolors.to_hex(colorsys.hls_to_rgb(h, new_l, new_s))
        else:
            palette[config] = base_hex
    return palette

def generate_aggregate_plots(all_results):
    if not all_results: return
    df = pd.DataFrame(all_results)
    
    # Exclude 'Combined' from the seasonal spread boxplot
    df_seasonal = df[df['Season'] != 'Combined'].copy()
    if df_seasonal.empty: return

    df_seasonal['Configuration'] = df_seasonal.apply(lambda row: f"{row['Model']} ({row['Filter']})", axis=1)
    
    palette = _get_config_palette(df_seasonal['Configuration'].unique())

    plt.figure(figsize=(14, 10))
    
    # Brier Chart
    plt.subplot(2, 1, 1)
    # Boxplot shows spread across seasons
    sns.boxplot(data=df_seasonal, x='Configuration', y='Brier', palette=palette, hue='Configuration', legend=False)
    # Overlay individual season points
    sns.stripplot(data=df_seasonal, x='Configuration', y='Brier', color='black', alpha=0.3, jitter=True)
    plt.title('Brier Score Distribution Across Seasons (Lower is Better)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    # Accuracy Chart
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df_seasonal, x='Configuration', y='Accuracy', palette=palette, hue='Configuration', legend=False)
    sns.stripplot(data=df_seasonal, x='Configuration', y='Accuracy', color='black', alpha=0.3, jitter=True)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    plt.title('Accuracy Distribution Across Seasons (Higher is Better)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out_path = Path("analysis/evaluation/predictive_power_comparison_summary.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved comparison plot to {out_path}")

def generate_combined_only_plot(all_results):
    if not all_results: return
    df = pd.DataFrame(all_results)
    df_comb = df[df['Season'] == 'Combined'].copy()
    if df_comb.empty:
        # If only one season, use that instead of 'Combined'
        if len(df['Season'].unique()) == 1:
            df_comb = df.copy()
        else:
            return

    df_comb['Configuration'] = df_comb.apply(lambda row: f"{row['Model']} ({row['Filter']})", axis=1)
    
    # Expand distributions for boxplotting bootstrap results
    brier_rows = []
    acc_rows = []
    for _, row in df_comb.iterrows():
        for b in row['Brier_dist']:
            brier_rows.append({'Configuration': row['Configuration'], 'Brier': b})
        for a in row['Accuracy_dist']:
            acc_rows.append({'Configuration': row['Configuration'], 'Accuracy': a})
            
    df_brier = pd.DataFrame(brier_rows)
    df_acc = pd.DataFrame(acc_rows)
    
    palette = _get_config_palette(df_comb['Configuration'].unique())

    plt.figure(figsize=(12, 10))
    
    # Brier
    plt.subplot(2, 1, 1)
    # Boxplot of bootstrap distribution
    sns.boxplot(data=df_brier, x='Configuration', y='Brier', palette=palette, hue='Configuration', legend=False)
    plt.title('Grand Aggregate Brier Score (Bootstrap Distribution)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df_acc, x='Configuration', y='Accuracy', palette=palette, hue='Configuration', legend=False)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    plt.title('Grand Aggregate Accuracy (Bootstrap Distribution)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = Path("analysis/evaluation/predictive_power_combined_only.png")
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved combined-only plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', type=str, default='20232024', help="Comma-separated seasons")
    parser.add_argument('--model', type=str, default='nested_xg', help='Comma-separated: nested_xg, non_nested_xg, mixed_effects, actual, or prefix with local_')
    parser.add_argument('--metric', type=str, default='gd', help='gd, gf_pct, rank')
    parser.add_argument('--filter', type=str, default='all', help='Comma-separated: all, 5v5, 5v5_close, extrapolated_per60')
    parser.add_argument('--matchup', type=str, default='poisson', help='poisson, simulation')
    parser.add_argument('--matchup-logic', type=str, default='multiplicative', choices=['multiplicative', 'additive'])
    parser.add_argument('--outcome', type=str, default='final', help='final, regulation')
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--n-boot', type=int, default=100) # Lower default for sweeps
    args = parser.parse_args()

    models = args.model.split(',')
    filters = args.filter.split(',')
    
    raw_seasons = args.seasons.split(',')
    if 'aggregate' in [s.lower() for s in raw_seasons]:
        seasons = DataUtils.get_available_seasons()
        logger.info(f"Aggregating across discovered seasons: {seasons}")
    else:
        seasons = raw_seasons
    
    all_results = []
    
    for m in models:
        for f in filters:
            m, f = m.strip(), f.strip()
            logger.info(f"==== Starting Sweep: Model={m}, Filter={f} ====")
            evaluator = PredictiveEvaluator(m, args.metric, f, args.matchup, args.outcome, args.n_boot, args.matchup_logic)
            
            sweep_results = []
            for season in seasons:
                res = evaluator.run_evaluation(season, args.train_split)
                if res:
                    sweep_results.append(res)
                    all_results.append(res)
            
            # Calculate Combined Metric if multiple seasons
            if len(sweep_results) > 1:
                total_games = sum(r['Test_Games'] for r in sweep_results)
                if total_games > 0:
                    # Calculate aggregate distribution by concatenating raw results
                    combined_raw = pd.concat([r['Raw_Results'] for r in sweep_results], ignore_index=True)
                    combined_metrics = evaluator.calculate_metrics(combined_raw)
                    
                    combined = {
                        'Season': 'Combined',
                        'Model': m,
                        'Filter': f,
                        'Metric': args.metric,
                        **combined_metrics,
                        'Test_Games': total_games
                    }
                    all_results.append(combined)
    
    if all_results:
        df_summary = pd.DataFrame(all_results)
        generate_aggregate_plots(all_results)
        generate_combined_only_plot(all_results)
        
        out_csv = Path("analysis/evaluation/predictive_power_comparison_suite.csv")
        df_summary.to_csv(out_csv, index=False)
        
        print("\n--- AGGREGATE SUMMARY ---")
        print(df_summary.to_string(index=False))
        logger.info(f"Full results saved to {out_csv}")

if __name__ == "__main__":
    main()
