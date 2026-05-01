"""scripts/evaluate_predictive_power.py

Overhauled predictive power evaluation routine.
Flexible and modular structure for assessing xG models and team performance metrics.

Available Options:
-----------------
--model:  nested_xg, non_nested_xg, mixed_effects_nested, mixed_effects_non_nested, 
           actual, moneypuck.
          Prefix with 'local_' to train on the specific season's training split 
          (e.g., local_nested_xg).
--filter: all (default), 5v5, 5v5_close (score diff <= 1), extrapolated_per60.
--metric: gd (Goal Difference, default), gf_pct (Goals For %), rank (Team Rankings).
--seasons: Comma-separated list (e.g., 20222023,20232024) or 'aggregate' to 
           auto-discover all seasons and calculate weighted combined results. 
           Can also specify '20202021+' to auto-include all modern era seasons.

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

4) Predict rest of season 20232024 cumulative stats using 5v5 training (70% split):
   python scripts/evaluate_predictive_power.py --predict-season --seasons 20232024 --filter 5v5 --train-split 0.7

5) Predict end of season totals, comparing 5v5 ability vs all-situations cumulative stats:
   python scripts/evaluate_predictive_power.py --predict-season --prediction-mode end_of_season --filter 5v5 --cumulative-filter all --seasons 20222023,20232024

6) Large-scale stability study for all modern era models:
   python scripts/evaluate_predictive_power.py --reps 500 --per-season --hockey-graphs \
     --filter all --seasons 20202021+ --model nested_xg,non_nested_xg,mixed_effects_nested,actual

7) Compare Nested vs Non-Nested mixed effects performance:
   python scripts/evaluate_predictive_power.py --seasons aggregate \
     --model mixed_effects_nested,mixed_effects_non_nested --filter all

8) Full Modern Era Stability Sweep with Mixed Effects models:
   python scripts/evaluate_predictive_power.py --seasons 20202021+ --model nested,non_nested,mixed_effects_nested,mixed_effects_non_nested,actual --filter all --hockey-graphs --reps 1000

9) Boss command to look at per-game accuracy across all primary models and filters:
   python scripts/evaluate_predictive_power.py --seasons 20202021+ \
     --model nested_xg,non_nested_xg,mixed_effects_nested,mixed_effects_non_nested,actual \
     --filter all,5v5 --n-boot 100 --parallel --n-jobs -1

10) Comprehensive Comparison: Nested vs Non-Nested (XGBoost & GLM) vs Actual Goals:
   python scripts/evaluate_predictive_power.py --seasons 20202021+ \
     --model xgboost_nested,xgboost_non_nested,xgboost_alternate,nested,non_nested,actual \
     --filter all --n-boot 100 --parallel --n-jobs -1

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
from puck import moneypuck, data_pipeline, fit_xgboost_nested, fit_xgboost_non_nested
from scripts import plot_predictive_power

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Utilities ---

class DataUtils:
    @staticmethod
    def load_season_data(season, apply_arena_adjustments=True):
        """Load and preprocess season data."""
        logger.info(f"Loading data for {season} (Arena Adj: {apply_arena_adjustments})...")
        csv_path = analyze.locate_season_csv(season)
        df = pd.read_csv(csv_path)
        
        # Pre-process for models
        # Alignment: Use impute_alpha=0.2 to match training (instead of default 0.5)
        df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=True, 
            apply_arena_adjustments=apply_arena_adjustments, 
            apply_bio_enrichment=True, apply_filtering=True,
            impute_alpha=0.2
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
                
                # Shootout Tie Fix: Ensure no 0.5 ties in 'final' outcomes unless data is missing
                # If goals are tied after counting all events, we might have missed the SO winner
                if home_goals_final == away_goals_final:
                    # Look for shootout goal markers if they exist
                    if 'period_type' in group.columns:
                        so_goals = group[(group['period'] > 4) | (group['period_type'].str.lower() == 'shootout')] # type: ignore
                    else:
                        so_goals = group[group['period'] > 4]
                        
                    if not so_goals.empty:
                        # Find which team has the most SO goals (usually one team is awarded the "deciding" goal)
                        home_so_wins = len(so_goals[so_goals['team_id'] == group['home_id']])
                        away_so_wins = len(so_goals[so_goals['team_id'] == group['away_id']])
                        if home_so_wins > away_so_wins: home_goals_final += 1
                        elif away_so_wins > home_so_wins: away_goals_final += 1
                
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
        data_dir = Path(analyze.puck_config.DATA_DIR)
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
        # Improved cache key: use a hash of the training game IDs for robustness
        # if train_df is provided and we expect a re-fit (local or mixed effects)
        train_id = None
        if train_df is not None:
            if is_local or model_name.startswith('mixed_effects'):
                # Sort for stability
                train_id = hash(tuple(sorted(train_df['game_id'].unique())))
            else:
                train_id = id(train_df) 

        cache_key = (model_name, is_local, train_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if model_name.startswith('mixed_effects') and not is_local:
            # "Global" mixed effects uses a global GLM as a fixed-effect base
            # but MUST fit team-specific intercepts on the current train_df.
            if train_df is None:
                logger.warning("train_df is None, cannot fit mixed effects intercepts.")
                return None
            
            # Determine base model type (default to XGBoost unless 'glm' is in name)
            if 'glm' in model_name:
                base_type = 'non_nested' if 'non_nested' in model_name else 'nested'
            else:
                base_type = 'xgboost_non_nested' if 'non_nested' in model_name else 'xgboost_nested'
                
            logger.info(f"Fitting mixed effects intercepts using {base_type} base...")
            
            base = self._load_global_model(base_type)
            if base is None:
                return None
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
            'xgboost_nested': os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_modern_era.joblib'),
            'nested_xg': os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_modern_era.joblib'),
            'nested': os.path.join('analysis', 'xgs', 'xg_model_nested_tensor_modern_era.joblib'),
            'xgboost_non_nested': os.path.join('analysis', 'xgs', 'xg_model_xgboost_non_nested_modern_era.joblib'),
            'non_nested_xg': os.path.join('analysis', 'xgs', 'xg_model_xgboost_non_nested_modern_era.joblib'),
            'non_nested': os.path.join('analysis', 'xgs', 'xg_model_non_nested_tensor_modern_era.joblib'),
            'xgboost_tensor': os.path.join('analysis', 'xgs', 'xg_model_xgboost_tensor_modern_era.joblib')
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
        
        if model_name in ['xgboost_nested', 'nested_xg']:
            model = fit_xgboost_nested.XGBNestedXGClassifier(features=feature_list)
            model.fit(train_df)
        elif model_name == 'nested':
            model = fit_glm_nested.NestedGLM(features=feature_list, use_splines=True, enable_marginalization=True)
            model.fit(train_df)
        elif model_name in ['xgboost_non_nested', 'non_nested_xg']:
            model = fit_xgboost_non_nested.XGBNonNestedXGClassifier(features=feature_list)
            model.fit(train_df)
        elif model_name == 'non_nested':
            model = fit_glm.NonNestedGLM(features=feature_list, use_splines=True, enable_marginalization=True)
            model.fit(train_df[train_df['event'] != 'blocked-shot'])
        elif model_name == 'xgboost_tensor':
            from puck import fit_xgboost_tensor
            model = fit_xgboost_tensor.XGBTensorXGClassifier(features=feature_list)
            model.fit(train_df)
        elif model_name.startswith('mixed_effects'):
            # Determine base model type (default to XGBoost unless 'glm' is in name)
            if 'glm' in model_name:
                base_type = 'non_nested' if 'non_nested' in model_name else 'nested'
            else:
                base_type = 'xgboost_non_nested' if 'non_nested' in model_name else 'xgboost_nested'
                
            base = self._train_local_model(base_type, train_df)
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
        is_local = 'local' in model_name
        pure_model_name = model_name.replace('local_', '')
        
        if pure_model_name != 'actual':
            model = model_registry.get_model(pure_model_name, train_df, is_local=is_local)
            if model:
                df = df.copy()
                # BUG 2 Fix: Nested models should score ALL events because they handle blocks internally.
                # Only non-nested GLM/XGBoost models should be filtered to outcome events.
                if pure_model_name.startswith('mixed_effects') or 'nested' in pure_model_name.lower():
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
        h_dist = poisson.pmf(np.arange(max_g), h_exp)
        a_dist = poisson.pmf(np.arange(max_g), a_exp)
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


class SeasonSimulator:
    """Monte Carlo simulator for season cumulative statistics."""
    def __init__(self, matchup_engine, seed=42):
        self.matchup_engine = matchup_engine
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def simulate_remaining_season(self, test_sched, team_abilities, league_avg, n_sims=1000, outcome_type='final'):
        teams = set(test_sched['home_team']).union(set(test_sched['away_team']))
        
        sim_wins = {t: np.zeros(n_sims) for t in teams}
        sim_gd = {t: np.zeros(n_sims) for t in teams}
        sim_xg_f = {t: 0.0 for t in teams}
        sim_xg_a = {t: 0.0 for t in teams}
        
        game_expectations = []
        for _, row in test_sched.iterrows():
            h, a = row['home_team'], row['away_team']
            h_stats = team_abilities.get(h, {'for': league_avg, 'ag': league_avg})
            a_stats = team_abilities.get(a, {'for': league_avg, 'ag': league_avg})
            
            if self.matchup_engine.logic_type == 'multiplicative':
                h_exp = (h_stats['for'] * a_stats['ag']) / league_avg if league_avg > 0 else 0.0
                a_exp = (a_stats['for'] * h_stats['ag']) / league_avg if league_avg > 0 else 0.0
            else:
                h_gd_val = h_stats['for'] - h_stats['ag']
                a_gd_val = a_stats['for'] - a_stats['ag']
                h_exp = league_avg + (h_gd_val - a_gd_val) / 2
                a_exp = league_avg + (a_gd_val - h_gd_val) / 2
            
            h_exp = max(0.1, h_exp)
            a_exp = max(0.1, a_exp)
            game_expectations.append((h, a, h_exp, a_exp))
            
            # Aggregate xG
            sim_xg_f[h] += h_exp
            sim_xg_a[h] += a_exp
            sim_xg_f[a] += a_exp
            sim_xg_a[a] += h_exp

        for i in range(n_sims):
            for h, a, h_exp, a_exp in game_expectations:
                h_goals = self.rng.poisson(h_exp)
                a_goals = self.rng.poisson(a_exp)
                
                if h_goals > a_goals:
                    sim_wins[h][i] += 1
                elif a_goals > h_goals:
                    sim_wins[a][i] += 1
                else:
                    if outcome_type == 'final':
                        if self.rng.random() > 0.5: sim_wins[h][i] += 1
                        else: sim_wins[a][i] += 1
                    else:
                        sim_wins[h][i] += 0.5
                        sim_wins[a][i] += 0.5
                
                sim_gd[h][i] += (h_goals - a_goals)
                sim_gd[a][i] += (a_goals - h_goals)
        
        results = {}
        for t in teams:
            results[t] = {
                'pred_wins_mean': np.mean(sim_wins[t]),
                'pred_wins_std': np.std(sim_wins[t]),
                'pred_wins_ci': (np.percentile(sim_wins[t], 2.5), np.percentile(sim_wins[t], 97.5)),
                'pred_gd_mean': np.mean(sim_gd[t]),
                'pred_gd_std': np.std(sim_gd[t]),
                'pred_gd_ci': (np.percentile(sim_gd[t], 2.5), np.percentile(sim_gd[t], 97.5)),
                'pred_xg_diff': sim_xg_f[t] - sim_xg_a[t]
            }
        return results

# --- Orchestration ---

class PredictiveEvaluator:
    def __init__(self, model_name, metric_type, filter_type, matchup_type='poisson', 
                 outcome_type='final', n_boot=100, matchup_logic='multiplicative', 
                 n_jobs=1, apply_arena_adjustments=True, seed=42, no_dashboards=False):
        self.model_registry = ModelRegistry()
        self.summarizer = TeamAbilitySummarizer(metric_type, filter_type)
        
        self.model_name = model_name
        self.metric_type = metric_type
        self.filter_type = filter_type
        self.matchup_type = matchup_type
        self.outcome_type = outcome_type
        self.n_boot = n_boot
        self.matchup_logic = matchup_logic
        self.n_jobs = n_jobs
        self.apply_arena_adjustments = apply_arena_adjustments
        self.seed = seed
        self.no_dashboards = no_dashboards

        # Engine Selection
        # Note: SimulationMatchupEngine removed as it was a Poisson fallback
        self.matchup_engine = PoissonMatchupEngine(logic_type=matchup_logic)

    def run_evaluation(self, season, train_split=0.7, split_method='random', n_reps=1):
        df = DataUtils.load_season_data(season, apply_arena_adjustments=self.apply_arena_adjustments)
        sched_df = DataUtils.process_schedule(df)
        all_gids = np.array(sched_df['game_id'].values, dtype=int)
        total_games = len(sched_df)
        n_train = int(total_games * train_split)
        
        if n_train >= total_games and split_method == 'chronological':
            logger.warning("Train split >= 1.0, cannot evaluate future games.")
            return None

        rep_metrics = []
        all_rep_results = []
        
        logger.info(f"Season {season}: Running {n_reps} reps using {split_method} split...")
        
        def _run_single_rep(r):
            if split_method == 'random':
                # Random sample of games for training
                # Use a specific seed per rep for reproducibility in parallel
                rng = np.random.default_rng(seed=42 + r)
                train_gids_rep = rng.choice(all_gids, n_train, replace=False)
                test_gids_rep = np.array([g for g in all_gids if g not in train_gids_rep])
            else:
                # Chronological split
                train_gids_rep = all_gids[:n_train]
                test_gids_rep = all_gids[n_train:]
            
            # Sub-sets for this rep
            train_df_rep = df[df['game_id'].isin(train_gids_rep)].copy()
            test_sched_rep = sched_df[sched_df['game_id'].isin(test_gids_rep)].copy()
            
            if len(test_sched_rep) == 0:
                return None

            # Summarize Ability (triggers re-fit if mixed effects or local model)
            abilities = self.summarizer.get_team_abilities(train_df_rep, self.model_name, self.model_registry)
            
            # THE FIX: Calculate League Average based on the CURRENT model's abilities
            # Using actual goals average for a potentially deflated/inflated xG model 
            # creates a scale mismatch in the Poisson denominator, hurting Brier scores.
            if abilities:
                league_avg_exp = np.mean([v['for'] for v in abilities.values()])
            else:
                # Fallback to empirical goals if no abilities (unlikely)
                train_df_filtered = self.summarizer._apply_filter(train_df_rep)
                # BUG 6 Fix: denominator should be 2 * number of games that passed the filter
                n_games_filtered = train_df_filtered['game_id'].nunique()
                league_avg_exp = (train_df_filtered['event'].str.lower() == 'goal').sum() / (2 * n_games_filtered) if n_games_filtered > 0 else 3.0
            
            rep_results_list = []
            for _, row in test_sched_rep.iterrows():
                p_hw = self.matchup_engine.predict_winner_prob(row['home_team'], row['away_team'], abilities, league_avg_exp, self.outcome_type)
                
                # Actual result (Fixed: handle ties explicitly for robustness)
                if self.outcome_type == 'final':
                    if row['home_goals_final'] > row['away_goals_final']:
                        actual = 1.0
                    elif row['home_goals_final'] < row['away_goals_final']:
                        actual = 0.0
                    else:
                        actual = 0.5 # Flexible default/fallback
                else:
                    if row['home_goals_reg'] > row['away_goals_reg']: actual = 1.0
                    elif row['away_goals_reg'] > row['home_goals_reg']: actual = 0.0
                    else: actual = 0.5
                    
                rep_results_list.append({'p': p_hw, 'y': actual})
                
            res_df = pd.DataFrame(rep_results_list)
            # Calculate metrics for THIS rep (disable nested bootstrap for speed)
            metrics = self.calculate_metrics(res_df, n_boot=0)
            return metrics, res_df, abilities

        if self.n_jobs != 1 and split_method == 'random':
            results = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(_run_single_rep)(r) for r in range(n_reps))
        else:
            results = [_run_single_rep(r) for r in range(n_reps)]
            
        for res in results:
            if res is None: continue
            metrics, res_df, abilities = res
            rep_metrics.append(metrics)
            all_rep_results.append(res_df)
            
            if split_method == 'chronological':
                break
                
        if not rep_metrics:
            return None
            
        # Aggregate across all reps
        briers = [m['Brier'] for m in rep_metrics]
        accs = [m['Accuracy'] for m in rep_metrics]
        
        summary = {
            'Brier': np.mean(briers),
            'Brier_lo': np.percentile(briers, 2.5) if len(briers) > 1 else np.mean(briers),
            'Brier_hi': np.percentile(briers, 97.5) if len(briers) > 1 else np.mean(briers),
            'Brier_dist': np.array(briers),
            'Accuracy': np.mean(accs),
            'Accuracy_lo': np.percentile(accs, 2.5) if len(accs) > 1 else np.mean(accs),
            'Accuracy_hi': np.percentile(accs, 97.5) if len(accs) > 1 else np.mean(accs),
            'Accuracy_dist': np.array(accs)
        }
        
        combined_raw = pd.concat(all_rep_results, ignore_index=True)
        
        logger.info(f"Season {season} Complete | Brier: {summary['Brier']:.4f} | Acc: {summary['Accuracy']:.4f}")
        
        # Rankings if requested (using the last rep's abilities as a representative sample)
        if self.metric_type == 'rank':
            self._output_rankings(season, abilities)

        eval_summary = {
            'Season': season,
            'Model': self.model_name,
            'Filter': self.filter_type,
            'Metric': self.metric_type,
            **summary,
            'Test_Games': len(combined_raw),
            'Raw_Results': combined_raw
        }
        
        # Baked-in Dashboard Generation
        pure_model_name = self.model_name.replace('local_', '')
        if pure_model_name != 'actual' and not self.no_dashboards:
            self._trigger_dashboard_generation(pure_model_name)

        return eval_summary

    def _trigger_dashboard_generation(self, model_name):
        """Automatically calls the appropriate dashboard script."""
        import subprocess
        
        # Map model name to dashboard script
        dashboard_map = {
            'nested_xg': ('scripts/nested_model_dashboard.py', 'analysis/xgs/xg_model_nested_tensor_20202021.joblib'),
            'nested': ('scripts/nested_model_dashboard.py', 'analysis/xgs/xg_model_nested_tensor_20202021.joblib'),
            'non_nested_xg': ('scripts/non_nested_model_dashboard.py', 'analysis/xgs/xg_model_non_nested_tensor_20202021.joblib'),
            'non_nested': ('scripts/non_nested_model_dashboard.py', 'analysis/xgs/xg_model_non_nested_tensor_20202021.joblib'),
            'xgboost_nested': ('scripts/xgboost_nested_model_dashboard.py', 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib'),
            'xgboost_non_nested': ('scripts/xgboost_non_nested_model_dashboard.py', 'analysis/xgs/xg_model_xgboost_non_nested_20202021.joblib'),
            'xgboost_tensor': ('scripts/xgboost_tensor_model_dashboard.py', 'analysis/xgs/xg_model_xgboost_tensor_modern_era.joblib')
        }
        
        if model_name in dashboard_map:
            script, model_path = dashboard_map[model_name]
            if os.path.exists(model_path):
                logger.info(f"Triggering dashboard update for {model_name}...")
                try:
                    subprocess.run([sys.executable, script, model_path], check=False, capture_output=True)
                except Exception as e:
                    logger.warning(f"Failed to generate dashboard for {model_name}: {e}")
            else:
                logger.warning(f"Could not find model at {model_path} for dashboard generation.")

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
    def run_season_prediction(self, season, train_split=0.7, n_sims=1000, cumulative_filter=None, prediction_mode='rest_of_season', split_method='chronological', split_reps=1):
        """Predicts season-level cumulative statistics with bootstrapping support."""
        df = DataUtils.load_season_data(season)
        sched_df = DataUtils.process_schedule(df)
        total_games = len(sched_df)
        n_train = int(total_games * train_split)
        
        if n_train >= total_games:
            logger.warning("Train split >= 1.0, cannot predict future games.")
            return None
            
        teams = pd.concat([sched_df['home_team'], sched_df['away_team']]).unique()
        target_filter = cumulative_filter if cumulative_filter else self.filter_type
        actual_summarizer = TeamAbilitySummarizer(metric_type='gd', filter_type=target_filter)
        
        is_local = 'local' in self.model_name
        pure_model_name = self.model_name.replace('local_', '')
        
        # Pre-calculate xG for ALL events once if model is not 'actual'
        df_filtered = actual_summarizer._apply_filter(df).copy()
        if pure_model_name != 'actual':
            model = self.model_registry.get_model(pure_model_name, df_filtered, is_local=is_local)
            if model is not None and not isinstance(model, str):
                mask = df_filtered['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
                df_filtered['eval_xg'] = 0.0
                if mask.any():
                    df_filtered.loc[mask, 'eval_xg'] = model.predict_proba(df_filtered[mask])[:, 1]
            else:
                df_filtered['eval_xg'] = (df_filtered['event'].str.lower() == 'goal').astype(float)
        else:
            df_filtered['eval_xg'] = (df_filtered['event'].str.lower() == 'goal').astype(float)
        
        # Pre-calculate game-team aggregates for speed
        # We need For/Against xG and Wins per game
        game_metrics = []
        for gid, group in df_filtered.groupby('game_id'):
            h, a = group['home_abb'].iloc[0], group['away_abb'].iloc[0]
            # All situations or filtered situations?
            # actual_summarizer already filtered df_filtered.
            
            # xG For / Against
            h_xg_f = group[group['team_id'] == group['home_id']]['eval_xg'].sum()
            a_xg_f = group[group['team_id'] == group['away_id']]['eval_xg'].sum()
            
            # Goals for Wins (from sched_df which has final outcomes)
            match = sched_df[sched_df['game_id'] == gid].iloc[0]
            h_win = 1 if match['home_goals_final'] > match['away_goals_final'] else 0
            a_win = 1 if match['away_goals_final'] > match['home_goals_final'] else 0
            
            # Goals For (empirical)
            h_gf = (group[(group['team_id'] == group['home_id']) & (group['event'].str.lower() == 'goal')]).shape[0]
            a_gf = (group[(group['team_id'] == group['away_id']) & (group['event'].str.lower() == 'goal')]).shape[0]
            
            game_metrics.append({
                'game_id': gid,
                'team': h,
                'opp': a,
                'xg_f': float(h_xg_f),
                'xg_a': float(a_xg_f),
                'gf': float(h_gf),
                'ga': float(a_gf),
                'win': h_win
            })
            game_metrics.append({
                'game_id': gid,
                'team': a,
                'opp': h,
                'xg_f': float(a_xg_f),
                'xg_a': float(h_xg_f),
                'gf': float(a_gf),
                'ga': float(h_gf),
                'win': a_win
            })
            
        gm_df = pd.DataFrame(game_metrics)
        all_gids = sched_df['game_id'].values
        
        rep_results = []
        
        # M3 Fix: Use seeded RNG for reproducibility
        rng = np.random.default_rng(seed=self.seed)
        
        for rep in range(split_reps):
            if split_method == 'random':
                train_gids = rng.choice(list(all_gids), n_train, replace=False)
                test_gids = np.array([g for g in all_gids if g not in train_gids])
            else:
                train_gids = all_gids[:n_train]
                test_gids = all_gids[n_train:]
            
            train_gm = gm_df[gm_df['game_id'].isin(train_gids)]
            test_gm = gm_df[gm_df['game_id'].isin(test_gids)]
            
            rep_stats = []
            for t in teams:
                t_tr = train_gm[train_gm['team'] == t]
                t_te = test_gm[test_gm['team'] == t]
                
                n_tr = len(t_tr)
                n_te = len(t_te)
                
                tr_xg_f = t_tr['xg_f'].sum()
                tr_xg_a = t_tr['xg_a'].sum()
                tr_gf = t_tr['gf'].sum()
                tr_ga = t_tr['ga'].sum()
                tr_wins = t_tr['win'].sum()
                
                te_xg_f = t_te['xg_f'].sum()
                te_xg_a = t_te['xg_a'].sum()
                te_gf = t_te['gf'].sum()
                te_ga = t_te['ga'].sum()
                te_wins = t_te['win'].sum()
                
                # Predictor (from train)
                predictor_xg_pg = (tr_xg_f - tr_xg_a) / n_tr if n_tr > 0 else 0.0
                predictor_gd_pg = (tr_gf - tr_ga) / n_tr if n_tr > 0 else 0.0
                
                # Outcome (from test)
                outcome_gd_pg = (te_gf - te_ga) / n_te if n_te > 0 else 0.0
                outcome_wins_pg = te_wins / n_te if n_te > 0 else 0.0
                
                rep_stats.append({
                    'Team': t,
                    'train_xg_diff_pg': predictor_xg_pg,
                    'train_gd_pg': predictor_gd_pg,
                    'test_gd_pg': outcome_gd_pg,
                    'test_wins_pg': outcome_wins_pg,
                    'test_gd': te_gf - te_ga,
                    'test_wins': te_wins,
                    'test_xg_diff': te_xg_f - te_xg_a,
                    'train_gd': tr_gf - tr_ga,
                    'train_xg_diff': tr_xg_f - tr_xg_a,
                    'train_wins': tr_wins,
                    'test_games': n_te,
                    'train_games': n_tr
                })
            
            rep_df = pd.DataFrame(rep_stats)
            
            # Calculate R2 for this rep
            # Use appropriate predictor column
            p_col = 'train_xg_diff_pg' if 'xg' in self.model_name.lower() else 'train_gd_pg'
            
            if len(rep_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(rep_df[p_col], rep_df['test_gd_pg'])
                r2 = r_value**2
            else:
                r2 = 0.0
                
            rep_results.append({
                'rep': rep,
                'r2': r2,
                'stats': rep_df
            })
            
            if split_method == 'chronological': break # One rep is enough
            
        # Aggregate bootstrapping results
        final_r2s = [float(r['r2']) for r in rep_results if isinstance(r, dict) and 'r2' in r] # type: ignore
        mean_r2 = float(np.mean(final_r2s)) if final_r2s else 0.0
        logger.info(f"Bootstrapping complete ({len(final_r2s)} reps). Mean R²: {mean_r2:.4f}")
        
        # USE THE FIRST REPETITION'S STATS FOR PLOTTING (to maintain realistic scatter)
        # But attach the mean results as attributes
        first_stats = rep_results[0]['stats']
        plotting_df = first_stats.copy() if hasattr(first_stats, 'copy') else pd.DataFrame(first_stats) # type: ignore
        
        # Re-add metadata
        plotting_df['Season'] = season
        plotting_df['Model'] = self.model_name
        plotting_df['Filter'] = self.filter_type
        plotting_df['CumFilter'] = target_filter
        plotting_df['PredictionMode'] = prediction_mode
        plotting_df['SplitMethod'] = split_method
        plotting_df.attrs['MeanR2'] = mean_r2
        plotting_df.attrs['r2_dist'] = final_r2s
        
        # Rename for common plotting logic
        rename_dict = {
            'test_gd': 'act_gd',
            'test_wins': 'act_wins',
            'test_xg_diff': 'act_xg_diff'
        }
        plotting_df = plotting_df.rename(columns=rename_dict)
        
        for col in ['pred_wins_mean', 'pred_wins_std', 'pred_gd_mean', 'pred_gd_std', 'pred_xg_diff']:
            plotting_df[col] = 0.0
        
        return plotting_df
            
    def run_hockey_graphs_stability(self, seasons, intervals=[10, 20, 30, 40, 50, 60, 70], reps=1000, per_season=False, hg_metric='pct'):
        """
        Replicates Hockey-Graphs methodology:
        - Select sample size X.
        - Bootstrap 1000 times:
            - Split each team-season into X games (A) and rest (B).
            - Pool all team-seasons across all seasons.
            - Calculate correlation between Metric A and Goals B.
        - Aggregate using Fisher-Z transformation.
        """
        logger.info(f"Starting Hockey-Graphs Stability Study across {len(seasons)} seasons...")
        hg_attrs = {}
        
        # 1. Load and pre-process all seasons
        all_season_gms = {}
        for season in seasons:
            df = DataUtils.load_season_data(season)
            sched_df = DataUtils.process_schedule(df)
            
            is_local = 'local' in self.model_name
            pure_model_name = self.model_name.replace('local_', '')
            
            # Pre-calculate xG for ALL events once using global model
            df_filtered = self.summarizer._apply_filter(df).copy()
            if pure_model_name != 'actual':
                model = self.model_registry.get_model(pure_model_name, df_filtered, is_local=is_local)
                if model is not None and not isinstance(model, str):
                    mask = df_filtered['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
                    df_filtered['eval_xg'] = 0.0
                    if mask.any():
                        df_filtered.loc[mask, 'eval_xg'] = model.predict_proba(df_filtered[mask])[:, 1]
                else:
                    df_filtered['eval_xg'] = (df_filtered['event'].str.lower() == 'goal').astype(float)
            else:
                df_filtered['eval_xg'] = (df_filtered['event'].str.lower() == 'goal').astype(float)
            
            # Aggregate to game-team metrics
            game_metrics = []
            for gid, group in df_filtered.groupby('game_id'):
                if group.empty: continue
                h = group['home_abb'].iloc[0]
                a = group['away_abb'].iloc[0]
                h_id = group['home_id'].iloc[0]
                a_id = group['away_id'].iloc[0]
                
                # Attribute xG and Goals to for/against per team
                mask_h = group['team_id'] == h_id
                mask_a = group['team_id'] == a_id
                mask_goal = group['event'].str.lower() == 'goal'
                
                h_xg_f = group[mask_h]['eval_xg'].sum()
                a_xg_f = group[mask_a]['eval_xg'].sum()
                h_gf = group[mask_h & mask_goal].shape[0]
                a_gf = group[mask_a & mask_goal].shape[0]
                
                game_metrics.append({'game_id': gid, 'team': h, 'xg_f': float(h_xg_f), 'xg_a': float(a_xg_f), 'gf': float(h_gf), 'ga': float(a_gf)})
                game_metrics.append({'game_id': gid, 'team': a, 'xg_f': float(a_xg_f), 'xg_a': float(h_xg_f), 'gf': float(a_gf), 'ga': float(h_gf)})
            
            gm_df = pd.DataFrame(game_metrics)
            
            # Map data to teams for fast lookup using NumPy arrays
            team_data = {}
            for team, group in gm_df.groupby('team'):
                team_data[team] = {
                    'xg_f': group['xg_f'].values,
                    'xg_a': group['xg_a'].values,
                    'gf': group['gf'].values,
                    'ga': group['ga'].values,
                    'game_ids': group['game_id'].values,
                    'count': len(group)
                }
            
            # Extract list of games with participant teams for this season
            season_games = []
            for gid, group in sched_df.iterrows():
                season_games.append({
                    'game_id': group['game_id'],
                    'home_team': group['home_team'],
                    'away_team': group['away_team'],
                    'home_goals_final': group['home_goals_final'],
                    'away_goals_final': group['away_goals_final'],
                    'actual': 1.0 if group['home_goals_final'] > group['away_goals_final'] else 0.0 if group['home_goals_final'] < group['away_goals_final'] else 0.5
                })
            
            all_season_gms[season] = {
                'team_data': team_data,
                'games': season_games
            }

        # Sampling Loop
        def fisher_z_mean(rs):
            if not rs: return 0.0
            rs_arr = np.clip(rs, -0.999, 0.999)
            mean_z = np.mean(np.arctanh(rs_arr))
            return np.tanh(mean_z)

        final_results = []
        
        # If per_season=True, we calculate for each season individually AND the aggregate pooled result.
        target_groups = [[s] for s in all_season_gms.keys()]
        if per_season:
            target_groups.append(list(all_season_gms.keys()))
        elif not per_season:
            target_groups = [list(all_season_gms.keys())]
        
        for group_seasons in target_groups:
            group_label = group_seasons[0] if len(group_seasons) == 1 else "Aggregate"
            logger.info(f"Running stability study for: {group_label}")
            
            hg_attrs = {}
            for X in intervals:
                logger.info(f"Processing interval: {X} games...")
                xg_rs, goal_rs = [], []
                brier_scores, accuracies = [], []
                
                for r in range(reps):
                    pooled_data = [] # (tr_xg_pct, tr_gf_pct, te_gf_pct)
                    rep_team_abilities = {}
                    rep_team_train_gids = {}
                    
                    total_tr_goals = 0
                    total_tr_games = 0
                    
                    # 1. First pass: Calculate abilities for all teams in this rep
                    for season in group_seasons:
                        curr_team_data = all_season_gms[season]['team_data']
                        for team, stats_dict in curr_team_data.items(): # type: ignore
                            n_total = stats_dict['count']
                            if n_total < X + 5: continue
                            
                            idx_all = np.random.permutation(n_total)
                            idx_a, idx_b = idx_all[:X], idx_all[X:]
                            
                            # Track training game IDs for predictive filtering
                            rep_team_train_gids[team] = set(stats_dict['game_ids'][idx_a])
                            
                            # Group A (Predictor)
                            tr_xgf, tr_xga = stats_dict['xg_f'][idx_a].sum(), stats_dict['xg_a'][idx_a].sum()
                            tr_gf, tr_ga = stats_dict['gf'][idx_a].sum(), stats_dict['ga'][idx_a].sum()
                            
                            # For mixed effects, apply shrinkage based on the model's L2 regularization.
                            # This simulates re-fitting the mixed effects model in each rep.
                            if 'mixed_effects' in self.model_name.lower():
                                # Pseudo-observations (K goals) based on l2_reg
                                # We try to get L2 from the model registry's cached model (without re-fitting)
                                is_local = 'local' in self.model_name
                                pure_model_name = self.model_name.replace('local_', '')
                                model_ref = self.model_registry._load_global_model(pure_model_name)
                                l2_reg = getattr(model_ref, 'l2_reg', 1.0)
                                
                                K = max(l2_reg, 5.0) 
                                
                                adj_f = (tr_gf + K) / (tr_xgf + K)
                                adj_a = (tr_ga + K) / (tr_xga + K)
                                
                                tr_xgf *= adj_f
                                tr_xga *= adj_a

                            if hg_metric == 'pct':
                                tr_xg_val = tr_xgf / (tr_xgf + tr_xga) if (tr_xgf + tr_xga) > 0 else 0.5
                                tr_gf_val_a = tr_gf / (tr_gf + tr_ga) if (tr_gf + tr_ga) > 0 else 0.5
                                # Store ability for predictive scoring
                                rep_team_abilities[team] = {'for': tr_xg_val, 'ag': 1.0 - tr_xg_val}
                            else: # diff
                                tr_xg_val = (tr_xgf - tr_xga) / X
                                tr_gf_val_a = (tr_gf - tr_ga) / X
                                # Store ability for predictive scoring
                                rep_team_abilities[team] = {'for': 0.5 + tr_xg_val/2, 'ag': 0.5 - tr_xg_val/2}
                            
                            # Group B (Outcome)
                            n_rest = int(n_total) - int(X)
                            te_gf_val, te_ga_val = stats_dict['gf'][idx_b].sum(), stats_dict['ga'][idx_b].sum()
                            
                            if hg_metric == 'pct':
                                te_gf_val_b = te_gf_val / (te_gf_val + te_ga_val) if (te_gf_val + te_ga_val) > 0 else 0.5
                            else: # diff
                                te_gf_val_b = (te_gf_val - te_ga_val) / n_rest
                            
                            pooled_data.append((tr_xg_val, tr_gf_val_a, te_gf_val_b))
                            total_tr_goals += tr_gf
                            total_tr_games += X
                    
                    if not pooled_data: continue
                    
                    rep_league_avg = (total_tr_goals / (2 * total_tr_games)) if total_tr_games > 0 else 3.0
                    
                    # 2. Second pass: Predictive power on games not used for training
                    rep_preds = []
                    for season in group_seasons:
                        season_games = all_season_gms[season]['games']
                        for game in season_games:
                            h, a = game['home_team'], game['away_team'] # type: ignore
                            
                            # Only predict if both teams have abilities calculated (and not used THIS game in training)
                            if h in rep_team_abilities and a in rep_team_abilities: # type: ignore
                                if game['game_id'] not in rep_team_train_gids.get(h, set()) and \
                                   game['game_id'] not in rep_team_train_gids.get(a, set()):
                                    p_hw = self.matchup_engine.predict_winner_prob(
                                        h, a, rep_team_abilities, rep_league_avg, self.outcome_type
                                    )
                                    rep_preds.append({'p': p_hw, 'y': game['actual']}) # type: ignore
                    
                    if rep_preds:
                        pred_df = pd.DataFrame(rep_preds)
                        brier = self._brier_fn(pred_df['y'].values, pred_df['p'].values)
                        acc = self._acc_fn(pred_df['y'].values, pred_df['p'].values)
                        brier_scores.append(brier)
                        accuracies.append(acc)

                    arr = np.array(pooled_data)
                    r_xg = np.corrcoef(arr[:, 0], arr[:, 2])[0, 1] if arr.shape[0] > 1 else np.nan
                    r_goal = np.corrcoef(arr[:, 1], arr[:, 2])[0, 1] if arr.shape[0] > 1 else np.nan
                    
                    if not np.isnan(r_xg): xg_rs.append(r_xg)
                    if not np.isnan(r_goal): goal_rs.append(r_goal)

                f_r_xg = fisher_z_mean(xg_rs)
                f_r_goal = fisher_z_mean(goal_rs)
                final_results.append({
                    'Season': group_label,
                    'Sample_Size': X,
                    'xG_r': f_r_xg, 'xG_r2': f_r_xg**2,
                    'Goals_r': f_r_goal, 'Goals_r2': f_r_goal**2,
                    'Accuracy': np.mean(accuracies) if accuracies else np.nan
                })
                # Store raw distributions for each interval in the DataFrame attributes
                if 'dist_map' not in hg_attrs: hg_attrs['dist_map'] = {}
                hg_attrs['dist_map'][X] = {'xg': xg_rs, 'goal': goal_rs}
            
        df_hg = pd.DataFrame(final_results)
        for k, v in hg_attrs.items():
            df_hg.attrs[k] = v
        return df_hg


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
        # BUG 5 Fix: Handle ties (0.5) correctly as "half-correct"
        # Standard accuracy: (p > 0.5) matches (y > 0.5)
        # Regulation ties (y=0.5) are tricky. We treat them as 0.5 if prediction is exactly 0.5, 
        # but realistically we just compare side.
        if isinstance(y, (np.ndarray, pd.Series)):
            # Vectorized version
            correct = ((p > 0.5) == (y > 0.5)).astype(float)
            # BUG 5 Fix: Skip ties in accuracy calculation to avoid sign-match bias
            tie_mask = (y == 0.5)
            if tie_mask.any() and not tie_mask.all():
                return float(np.mean(correct[~tie_mask]))
            return float(np.mean(correct))
        
        if y == 0.5: return 0.5 # Single tie case
        return float((p > 0.5) == (y > 0.5))

def _get_config_palette(configurations):
    """Creates a color palette where local/non-local models are linked."""
    palette = {}
    # Modern, professional base colors
    base_colors = {
        'mixed_effects_nested': '#1f77b4',     # Blue
        'nested_xg': '#aec7e8',                # Light Blue
        'nested': '#aec7e8',                   # Light Blue
        'mixed_effects_non_nested': '#2ca02c', # Green
        'non_nested_xg': '#98df8a',            # Light Green
        'non_nested': '#98df8a',               # Light Green
        'actual': '#d62728',                   # Black
        'moneypuck': '#9467bd'                 # Purple
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

def generate_aggregate_plots(all_results, filter_str='all'):
    if not all_results: return
    df = pd.DataFrame(all_results)
    
    # Exclude 'Combined' from the seasonal spread boxplot
    df_seasonal = df[df['Season'] != 'Combined'].copy()
    if df_seasonal.empty: return

    df_seasonal['Configuration'] = df_seasonal.apply(lambda row: f"{row['Model']} ({row['Filter']})", axis=1)
    
    # Explicit ordering for x-axis
    def get_order_key(config):
        model = config.split(' (')[0]
        order = {
            'mixed_effects_nested': 0,
            'nested_xg': 1, 'nested': 1,
            'mixed_effects_non_nested': 2,
            'non_nested_xg': 3, 'non_nested': 3,
            'actual': 4
        }
        return order.get(model, 10)
    
    order_configs = sorted(df_seasonal['Configuration'].unique(), key=get_order_key)
    palette = _get_config_palette(order_configs)

    plt.figure(figsize=(14, 10))
    
    # Brier Chart
    plt.subplot(2, 1, 1)
    # Boxplot shows spread across seasons
    sns.boxplot(data=df_seasonal, x='Configuration', y='Brier', palette=palette, hue='Configuration', order=order_configs, legend=False)
    # Overlay individual season points
    sns.stripplot(data=df_seasonal, x='Configuration', y='Brier', color='black', alpha=0.3, jitter=True)
    plt.title('Brier Score Distribution Across Seasons (Lower is Better)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    # Accuracy Chart
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df_seasonal, x='Configuration', y='Accuracy', palette=palette, hue='Configuration', order=order_configs, legend=False)
    sns.stripplot(data=df_seasonal, x='Configuration', y='Accuracy', color='black', alpha=0.3, jitter=True)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    plt.title('Accuracy Distribution Across Seasons (Higher is Better)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out_path = Path(f"analysis/evaluation/predictive_power_comparison_summary_{filter_str}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved comparison plot to {out_path}")

def generate_combined_only_plot(all_results, filter_str='all'):
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
    
    # Explicit ordering
    def get_order_key(config):
        model = config.split(' (')[0]
        order = {
            'mixed_effects_nested': 0,
            'nested_xg': 1, 'nested': 1,
            'mixed_effects_non_nested': 2,
            'non_nested_xg': 3, 'non_nested': 3,
            'actual': 4
        }
        return order.get(model, 10)
    
    order_configs = sorted(df_comb['Configuration'].unique(), key=get_order_key)
    palette = _get_config_palette(order_configs)

    plt.figure(figsize=(12, 10))
    
    # Brier
    plt.subplot(2, 1, 1)
    # Boxplot of bootstrap distribution
    sns.boxplot(data=df_brier, x='Configuration', y='Brier', palette=palette, hue='Configuration', order=order_configs, legend=False)
    plt.title('Grand Aggregate Brier Score (Bootstrap Distribution)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df_acc, x='Configuration', y='Accuracy', palette=palette, hue='Configuration', order=order_configs, legend=False)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    plt.title('Grand Aggregate Accuracy (Bootstrap Distribution)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(f"analysis/evaluation/predictive_power_combined_only_{filter_str}.png")
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved combined-only plot to {out_path}")

def generate_season_prediction_plots(results_df):
    if results_df.empty: return
    
    model_name = results_df['Model'].iloc[0]
    split_method = results_df['SplitMethod'].iloc[0] if 'SplitMethod' in results_df.columns else 'chronological'
    
    # Use xG Diff per game as predictor if it's an xG model, otherwise Goal Diff per game
    predictor_col = 'train_xg_diff_pg' if 'xg' in model_name.lower() else 'train_gd_pg'
    predictor_label = 'Training xG Diff/G' if 'xg' in model_name.lower() else 'Training Goal Diff/G'
    
    # If we have a distribution of R2 values, add a 4th plot
    r2_dist = results_df.attrs.get('r2_dist', [])
    n_plots = 4 if r2_dist and len(r2_dist) > 1 else 3
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))
    
    metrics = [
        ('act_wins', 'Actual Wins'),
        ('act_gd', 'Actual Goal Differential'),
        ('act_xg_diff', 'Actual xG Differential')
    ]
    
    for i, (act_col, label) in enumerate(metrics):
        ax = axes[i]
        x = results_df[predictor_col]
        y = results_df[act_col]
        
        # Scatter
        ax.scatter(x, y, alpha=0.6, edgecolors='w', s=100)
        
        # Regression Line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color='red', alpha=0.5, label=f'R²={r_value**2:.3f}')
        
        ax.set_title(f'{label} vs {predictor_label}')
        ax.set_xlabel(predictor_label)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    if n_plots == 4:
        ax = axes[3]
        sns.histplot(r2_dist, kde=True, ax=ax, color='green', alpha=0.4)
        mean_r2 = np.mean(r2_dist)
        ax.axvline(mean_r2, color='red', linestyle='--', label=f'Mean R²={mean_r2:.3f}')
        title_prefix = "Aggregate " if results_df['Season'].nunique() > 1 else ""
        ax.set_title(f'{title_prefix}Predictive Power Stability (R² Distribution)')
        ax.set_xlabel('R² value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    model_str = results_df['Model'].iloc[0]
    filter_str = results_df['Filter'].iloc[0]
    split_str = "random" if split_method == "random" else "chron"
    out_path = Path(f"analysis/evaluation/season_prediction_{model_str}_{filter_str}_{split_str}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved season prediction plot to {out_path}")

def generate_hockey_graphs_plots(hg_df, model_name, filter_type, metric='pct'):
    plt.figure(figsize=(12, 7))
    
    label_suffix = "Difference/G" if metric == 'diff' else "%"
    pred_label = f"xG {label_suffix}" if model_name != 'actual' else f"Actual Goal {label_suffix}"
    outcome_label = f"Future Goal {label_suffix}"
    
    unique_seasons = sorted([s for s in hg_df['Season'].unique() if s != "Aggregate"])
    num_seasons = len(unique_seasons)
    
    if num_seasons > 1:
        # Heat gradient: Cooler (Blue) for older -> Warmer (Red) for newer
        # RdYlBu_r provides a nice transition from Blue to Yellow to Red
        colors = plt.get_cmap('RdYlBu_r')(np.linspace(0.1, 0.9, num_seasons))
        palette = {s: c for s, c in zip(unique_seasons, colors)}
        
        # Add Aggregate if present in data
        if "Aggregate" in hg_df['Season'].unique():
            palette["Aggregate"] = "black"
            
        sns.lineplot(data=hg_df, x='Sample_Size', y='xG_r2', hue='Season', palette=palette, marker='o', alpha=0.8, linewidth=2.5)
        
        # If Aggregate exists, draw it prominently over others
        if "Aggregate" in hg_df['Season'].unique():
            agg_data = hg_df[hg_df['Season'] == "Aggregate"]
            label = f"{num_seasons}-Season Aggregate" if num_seasons > 0 else "Aggregate"
            plt.plot(agg_data['Sample_Size'], agg_data['xG_r2'], color='black', marker='D', linewidth=5, label=label, zorder=20)
            
        plt.title(f'Seasonal Stability Variance: {pred_label} vs {outcome_label} ({filter_type})', fontsize=16, fontweight='bold')
        plt.ylabel('R² (Predictive Power)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Season (Old -> New)")
    else:
        # Simplified aggregate plot
        plt.plot(hg_df['Sample_Size'], hg_df['xG_r2'], marker='o', label=f'{pred_label} vs {outcome_label} ({model_name})', color='#1f77b4', linewidth=3)
        plt.plot(hg_df['Sample_Size'], hg_df['Goals_r2'], marker='s', label=f'Actual Goal {label_suffix} vs {outcome_label}', color='#d62728', linewidth=3, linestyle='--')
        plt.title(f'Net Metric Reliability Curves: {pred_label} vs {outcome_label} ({filter_type})', fontsize=14)
        plt.ylabel(f'R² with Remaining Games ({outcome_label})', fontsize=12)
        plt.legend()
    
    plt.xlabel('Number of Games in Sample (Group A)', fontsize=12)
    plt.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    is_per_season = num_seasons > 1
    suffix = "_seasonal" if is_per_season else ""
    out_path = Path(f"analysis/evaluation/hockey_graphs_stability_{model_name}_{filter_type}{suffix}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved Hockey-Graphs plot to {out_path}")

    # Generate additional predictive plots if Accuracy/Brier are present
    if 'Accuracy' in hg_df.columns and 'Brier' in hg_df.columns:
        # 1. Accuracy Curve
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=hg_df, x='Sample_Size', y='Accuracy', hue='Season', marker='o', linewidth=2.5)
        if "Aggregate" in hg_df['Season'].unique():
            agg_data = hg_df[hg_df['Season'] == "Aggregate"]
            plt.plot(agg_data['Sample_Size'], agg_data['Accuracy'], color='black', marker='D', linewidth=4, label='Aggregate')
        plt.title(f'Predictive Accuracy vs Sample Size ({model_name}, {filter_type})', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Number of Training Games (N)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        acc_path = Path(f"analysis/evaluation/hockey_graphs_accuracy_{model_name}_{filter_type}.png")
        plt.savefig(acc_path, dpi=300)
        plt.close()

        # 2. Brier Score Curve
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=hg_df, x='Sample_Size', y='Brier', hue='Season', marker='o', linewidth=2.5)
        if "Aggregate" in hg_df['Season'].unique():
            agg_data = hg_df[hg_df['Season'] == "Aggregate"]
            plt.plot(agg_data['Sample_Size'], agg_data['Brier'], color='black', marker='D', linewidth=4, label='Aggregate')
        plt.title(f'Brier Score vs Sample Size ({model_name}, {filter_type})', fontsize=14, fontweight='bold')
        plt.ylabel('Brier Score (Lower is Better)', fontsize=12)
        plt.xlabel('Number of Training Games (N)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        brier_path = Path(f"analysis/evaluation/hockey_graphs_brier_{model_name}_{filter_type}.png")
        plt.savefig(brier_path, dpi=300)
        plt.close()

def generate_hockey_graphs_comparison_plot(hg_dfs, filter_type, metric='pct'):
    """
    Plots multiple model stability curves on the same chart for comparison.
    hg_dfs: Dictionary of {model_name: hg_df}
    """
    plt.figure(figsize=(12, 7))
    
    label_suffix = "Diff/G" if metric == 'diff' else "%"
    outcome_label = f"Future Goal {label_suffix}"
    
    # Standard colors for comparison
    colors = {
        'mixed_effects_nested': '#1f77b4',     # Blue
        'nested_xg': '#aec7e8',                # Light Blue
        'nested': '#aec7e8',                   # Light Blue
        'mixed_effects_non_nested': '#2ca02c', # Green
        'non_nested_xg': '#98df8a',            # Light Green
        'non_nested': '#98df8a',               # Light Green
        'actual': '#d62728',                   # Red
        'moneypuck': '#9467bd'                 # Purple
    }
    
    for i, (model_name, hg_df) in enumerate(hg_dfs.items()):
        # Focus on Aggregate if multiple seasons, otherwise use the only one
        if "Aggregate" in hg_df['Season'].unique():
            data = hg_df[hg_df['Season'] == "Aggregate"]
        else:
            data = hg_df
            
        color = colors.get(model_name, plt.get_cmap('tab10')(i))
        label = f"xG {label_suffix} vs {outcome_label} ({model_name})" if model_name != 'actual' else f"Actual Goal {label_suffix} vs {outcome_label}"
        marker = 'o' if model_name != 'actual' else 's'
        ls = '-' if model_name != 'actual' else '--'
        lw = 3
        
        plt.plot(data['Sample_Size'], data['xG_r2'], marker=marker, label=label, color=color, linewidth=lw, linestyle=ls)
        
        # If it's the first model (or specified), also plot the Goals baseline once
        # But wait, if model_name is 'actual', its xG_r2 IS the goals_r2 of the other models.
        # So we just plot xG_r2 for everyone.
        
    plt.title(f'Comparative Metric Reliability: Modern Era ({filter_type})', fontsize=16, fontweight='bold')
    plt.ylabel('R² (Predictive Power)', fontsize=12)
    plt.xlabel('Number of Games in Sample (Group A)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    model_count = len(hg_dfs)
    out_path = Path(f"analysis/evaluation/hockey_graphs_stability_comparison_{metric}_{filter_type}_{model_count}_models.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved Hockey-Graphs comparison plot to {out_path}")

    # Predictive Comparison Plots
    if all('Accuracy' in df.columns for df in hg_dfs.values()):
        plt.figure(figsize=(12, 7))
        for i, (m_name, df) in enumerate(hg_dfs.items()):
            data = df[df['Season'] == "Aggregate"] if "Aggregate" in df['Season'].unique() else df
            color = colors.get(m_name, plt.get_cmap('tab10')(i))
            plt.plot(data['Sample_Size'], data['Accuracy'], marker='o', label=m_name, color=color, linewidth=3)
        plt.title(f'Comparative Prediction Accuracy: Modern Era ({filter_type})', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Number of Training Games (N)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(f"analysis/evaluation/hockey_graphs_comparison_accuracy_{filter_type}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 7))
        for i, (m_name, df) in enumerate(hg_dfs.items()):
            data = df[df['Season'] == "Aggregate"] if "Aggregate" in df['Season'].unique() else df
            color = colors.get(m_name, plt.get_cmap('tab10')(i))
            plt.plot(data['Sample_Size'], data['Brier'], marker='o', label=m_name, color=color, linewidth=3)
        plt.title(f'Comparative Brier Score: Modern Era ({filter_type})', fontsize=16, fontweight='bold')
        plt.ylabel('Brier Score (Lower is Better)', fontsize=12)
        plt.xlabel('Number of Training Games (N)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(f"analysis/evaluation/hockey_graphs_comparison_brier_{filter_type}.png"), dpi=300)
        plt.close()

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
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for bootstrap repetitions')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (default -1 uses all cores)')
    parser.add_argument('--predict-season', action='store_true', help='If set, performs season-level cumulative statistics prediction')
    parser.add_argument('--cumulative-filter', type=str, default=None, help='Filter for cumulative statistics (wins/gd/xg). Defaults to same as --filter')
    parser.add_argument('--prediction-mode', type=str, default='rest_of_season', choices=['rest_of_season', 'end_of_season'], help='Predict rest of season or full season')
    parser.add_argument('--split-method', type=str, default='random', choices=['chronological', 'random'], help='Method for splitting season into train/test')
    parser.add_argument('--split-reps', type=int, default=None, help='Number of iterations for evaluation (full bootstrap). Defaults to --n-boot.')
    parser.add_argument('--reps', type=int, default=500, help='Number of bootstrap iterations for Hockey-Graphs study')
    parser.add_argument('--n-sims', type=int, default=1000, help='Number of Monte Carlo simulations for season prediction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-dashboards', action='store_true', help='Disable automatic dashboard generation/updates')
    # Hockey Graphs Study
    parser.add_argument('--hockey-graphs', action='store_true', help='Replicate Hockey-Graphs stability intervals analysis')
    parser.add_argument('--hg-metric', type=str, default='pct', choices=['pct', 'diff'], help='Metric for Hockey-Graphs study: pct (ratio) or diff (per-game difference)')
    parser.add_argument('--per-season', action='store_true', help='If set with --hockey-graphs, calculates stability per season')
    parser.add_argument('--no-arena-adj', action='store_true', help='Disable per-arena location bias adjustments')
    
    args = parser.parse_args()

    models = args.model.split(',')
    filters = args.filter.split(',')
    
    raw_seasons = args.seasons.split(',')
    if args.seasons == 'aggregate':
        seasons = DataUtils.get_available_seasons()
        logger.info(f"Aggregating across discovered seasons: {seasons}")
    elif args.seasons == '20202021+':
        all_seasons = DataUtils.get_available_seasons()
        seasons = [s for s in all_seasons if int(s) >= 20202021]
        logger.info(f"Aggregating across Modern Era (20202021+): {seasons}")
    else:
        seasons = raw_seasons
    
    all_results = []
    
    hockey_graphs_results = {}
    
    for m in models:
        for f in filters:
            m, f = m.strip(), f.strip()
            logger.info(f"==== Starting Sweep: Model={m}, Filter={f} ====")
            n_jobs = args.n_jobs if args.parallel else 1
            evaluator = PredictiveEvaluator(
                m, args.metric, f, args.matchup, args.outcome, args.n_boot, 
                args.matchup_logic, n_jobs=n_jobs, 
                apply_arena_adjustments=not args.no_arena_adj,
                seed=args.seed, no_dashboards=args.no_dashboards
            )
            
            if args.hockey_graphs:
                hg_df = evaluator.run_hockey_graphs_stability(seasons, reps=args.reps, per_season=args.per_season, hg_metric=args.hg_metric)
                out_path = Path(f"analysis/evaluation/hockey_graphs_stability_{args.hg_metric}_{m}_{f}.csv")
                hg_df.to_csv(out_path, index=False)
                generate_hockey_graphs_plots(hg_df, m, f, metric=args.hg_metric)
                
                # Collect for comparison
                hockey_graphs_results[m] = hg_df
                
                print(f"\n--- Hockey-Graphs Stability Study ({m}, {f}) ---")
                print(hg_df)
                continue

            sweep_results = []
            season_preds = []
            for season in seasons:
                if args.predict_season:
                    res = evaluator.run_season_prediction(
                        season, args.train_split, args.n_sims, 
                        args.cumulative_filter, args.prediction_mode,
                        split_method=args.split_method, split_reps=args.split_reps
                    )
                    if res is not None:
                        season_preds.append(res)
                else:
                    # Use split_reps if provided, otherwise fallback to n_boot for the full-lifecycle reps
                    n_reps = args.split_reps if args.split_reps is not None else args.n_boot
                    res = evaluator.run_evaluation(season, args.train_split, split_method=args.split_method, n_reps=n_reps)
                    if res:
                        sweep_results.append(res)
                        all_results.append(res)
            
            if args.predict_season and season_preds:
                all_season_df = pd.concat(season_preds, ignore_index=True)
                
                # Accumulate the R2 distributions from all seasons
                aggregate_r2_dist = []
                for res in season_preds:
                    if 'r2_dist' in res.attrs:
                        aggregate_r2_dist.extend(res.attrs['r2_dist'])
                
                if aggregate_r2_dist:
                    all_season_df.attrs['r2_dist'] = aggregate_r2_dist
                
                generate_season_prediction_plots(all_season_df)
                
                out_path = Path(f"analysis/evaluation/season_prediction_{m}_{f}.csv")
                all_season_df.to_csv(out_path, index=False)
                print(f"\n--- Season Prediction Summary ({m}, {f}) ---")
                print(all_season_df.drop(columns=['pred_wins_ci', 'pred_gd_ci'], errors='ignore').head())
                logger.info(f"Season prediction results saved to {out_path}")

            # Calculate Combined Metric if multiple seasons and NOT predict_season
            if not args.predict_season and len(sweep_results) > 1:
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
    
    # If we have multiple Hockey-Graphs results, generate comparison plot
    if args.hockey_graphs and len(hockey_graphs_results) > 1:
        # Assuming for now they all use the same filter (first one)
        generate_hockey_graphs_comparison_plot(hockey_graphs_results, filters[0], metric=args.hg_metric)
    
    if all_results:
        df_summary = pd.DataFrame(all_results)
        
        # Use first filter for filename if multiple filters were run (rarely happens in this script's flow)
        filter_str = filters[0] if filters else 'all'
        if args.no_arena_adj:
            filter_str += "_no_arena_adj"
        
        generate_aggregate_plots(all_results, filter_str=filter_str)
        generate_combined_only_plot(all_results, filter_str=filter_str)
        
        out_csv = Path(f"analysis/evaluation/predictive_power_comparison_suite_{filter_str}.csv")
        df_summary.to_csv(out_csv, index=False)
        
        print("\n--- AGGREGATE SUMMARY ---")
        print(df_summary.to_string(index=False))
        logger.info(f"Full results saved to {out_csv}")

if __name__ == "__main__":
    main()
