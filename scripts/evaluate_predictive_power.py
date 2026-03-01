"""scripts/evaluate_predictive_power.py

Evaluates the predictive power of Actual Goals, standard xG, and Mixed Effects xtG 
over N-game samples. Helps determine when the signal stabilizes.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from scipy.stats import poisson, spearmanr
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import mixed_effects
from puck import nhl_api
from puck import fit_glm_nested
from puck import features as feature_util
from scripts import plot_predictive_power

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_season_schedule(season):
    """Load schedule outcomes for the season to determine actual points."""
    logger.info(f"Loading official schedule for {season} from NHL API...")
    try:
        api_games = nhl_api.get_season('all', season, ['02'])
        season_schedule_dict = {}
        for g in api_games:
            gid = int(g.get('id') or g.get('gamePk') or g.get('gameID') or 0)
            if not gid: continue
            
            h_score = g.get('homeTeam', {}).get('score', 0)
            a_score = g.get('awayTeam', {}).get('score', 0)
            
            pd_type = g.get('periodDescriptor', {}).get('periodType', '')
            go_type = g.get('gameOutcome', {}).get('lastPeriodType', '')
            is_ot_so = pd_type in ['OT', 'SO'] or go_type in ['OT', 'SO']
            
            season_schedule_dict[gid] = {
                'home_goals': h_score,
                'away_goals': a_score,
                'is_ot_so': is_ot_so
            }
        return season_schedule_dict
    except Exception as e:
        logger.warning(f"Could not load NHL API schedule for {season}: {e}. Fallback to PxP heuristics.")
        return None

def process_schedule_from_events(df, season_schedule_dict=None):
    """
    Extracts a chronological list of games from the event dataframe.
    Returns a dataframe of games with home_team, away_team, home_goals, away_goals, is_ot_so, game_id.
    """
    logger.info("Extracting schedule from event data...")
    games = []
    
    for gid, group in df.groupby('game_id'):
        home_team = group['home_abb'].iloc[0]
        away_team = group['away_abb'].iloc[0]
        
        # Fallback Goals
        home_goals = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['home_id'])])
        away_goals = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == group['away_id'])])
        is_ot_so = group['period'].max() > 3
        
        if season_schedule_dict and gid in season_schedule_dict:
            actual = season_schedule_dict[gid]
            home_goals = actual['home_goals']
            away_goals = actual['away_goals']
            is_ot_so = actual['is_ot_so']
        
        games.append({
            'game_id': gid,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'is_ot_so': is_ot_so
        })
        
    sched_df = pd.DataFrame(games)
    sched_df = sched_df.sort_values('game_id').reset_index(drop=True)
    return sched_df

def get_team_game_counts(sched_df):
    """Given a schedule, assign a 'team_game_number' to each team for every game."""
    h_counts = {}
    a_counts = {}
    
    # We want a dictionary that tells us, for a given team, the list of their game_ids in order
    team_schedules = {}
    
    for _, row in sched_df.iterrows():
        h = row['home_team']
        a = row['away_team']
        gid = row['game_id']
        
        if h not in team_schedules: team_schedules[h] = []
        if a not in team_schedules: team_schedules[a] = []
        
        team_schedules[h].append(gid)
        team_schedules[a].append(gid)
        
    return team_schedules

def calculate_standings(sched_df):
    """Calculates points and points percentage for all teams in a schedule dataframe."""
    team_stats = {}
    
    for _, row in sched_df.iterrows():
        h = row['home_team']
        a = row['away_team']
        
        if h not in team_stats: team_stats[h] = {'points': 0, 'games': 0}
        if a not in team_stats: team_stats[a] = {'points': 0, 'games': 0}
        
        team_stats[h]['games'] += 1
        team_stats[a]['games'] += 1
        
        if row.get('is_ot_so', False):
            # If the game went to OT or SO, it was tied at the end of regulation
            team_stats[h]['points'] += 1
            team_stats[a]['points'] += 1
        elif row['home_goals'] > row['away_goals']:
            # Regulation home win
            team_stats[h]['points'] += 2
        elif row['away_goals'] > row['home_goals']:
            # Regulation away win
            team_stats[a]['points'] += 2
        else:
            # Tie (in case is_ot_so is missing somehow, fallback to goals)
            team_stats[h]['points'] += 1
            team_stats[a]['points'] += 1
            
    for t in team_stats:
        team_stats[t]['pts_pct'] = team_stats[t]['points'] / (team_stats[t]['games'] * 2) if team_stats[t]['games'] > 0 else 0
        
    return team_stats

def split_data(df, team_schedules, n_games):
    """
    Splits the event dataframe into Train (first N games for EACH team) 
    and Test (games where AT LEAST ONE team has played > N games).
    
    Wait, if a game is team A's 10th game but team B's 11th game, it goes in Test.
    Train only contains events from games that are <= N for BOTH teams.
    """
    train_game_ids = set()
    test_game_ids = set()
    
    all_game_ids = df['game_id'].unique()
    
    for gid in all_game_ids:
        # Find who is playing
        sample = df[df['game_id'] == gid].iloc[0]
        h = sample['home_abb']
        a = sample['away_abb']
        
        # What game number is this for home and away?
        if h in team_schedules and gid in team_schedules[h]:
            h_idx = team_schedules[h].index(gid) + 1
        else:
            h_idx = 999
            
        if a in team_schedules and gid in team_schedules[a]:
            a_idx = team_schedules[a].index(gid) + 1
        else:
            a_idx = 999
            
        if h_idx <= n_games and a_idx <= n_games:
            train_game_ids.add(gid)
        else:
            # We only evaluate games where BOTH teams have at least N games of history to make a prediction
            if h_idx > n_games and a_idx > n_games:
                test_game_ids.add(gid)
            
    train_df = df[df['game_id'].isin(train_game_ids)].copy()
    test_df = df[df['game_id'].isin(test_game_ids)].copy()
    
    return train_df, test_df

def extract_rates(train_df, state_mask=None):
    """Calculate Actual Goals and standard xG rates (per 60) for all teams in the train set."""
    if state_mask is not None:
        df = train_df[state_mask].copy()
    else:
        df = train_df.copy()
        
    teams = pd.concat([df['home_abb'], df['away_abb']]).unique()
    
    rates = {}
    for t in teams:
        rates[t] = {
            'gf60': 0.0, 'ga60': 0.0,
            'xgf60': 0.0, 'xga60': 0.0,
            'mp_xgf60': 0.0, 'mp_xga60': 0.0,
            'games': 0
        }
    
    for t in teams:
        team_games = df[(df['home_abb'] == t) | (df['away_abb'] == t)]['game_id'].nunique()
        if team_games == 0: continue
        
        # Goals for team t
        # A goal is for team t if they are the home team and team_id == home_id, OR if they are away team and team_id == away_id
        is_gf = (df['event'].str.lower() == 'goal') & (
            ((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | 
            ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))
        )
        is_ga = (df['event'].str.lower() == 'goal') & (
            ((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | 
            ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))
        )
        
        gf = len(df[is_gf])
        ga = len(df[is_ga])
        
        # Standard xG
        if 'xgs' in df.columns:
            xgf = df[((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))]['xgs'].sum()
            xga = df[((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))]['xgs'].sum()
        else:
            xgf, xga = 0, 0
            
        # MoneyPuck xG
        if 'mp_xG' in df.columns:
            mp_xgf = df[((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))]['mp_xG'].sum()
            mp_xga = df[((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))]['mp_xG'].sum()
        elif 'xG' in df.columns:
            # Maybe the column name was capitalized differently
            mp_xgf = df[((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))]['xG'].sum()
            mp_xga = df[((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))]['xG'].sum()
        else:
            mp_xgf, mp_xga = 0, 0
            
        # Local Nested xG
        if 'local_xgs' in df.columns:
            local_xgf = df[((df['home_abb'] == t) & (df['team_id'] == df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] == df['away_id']))]['local_xgs'].sum()
            local_xga = df[((df['home_abb'] == t) & (df['team_id'] != df['home_id'])) | ((df['away_abb'] == t) & (df['team_id'] != df['away_id']))]['local_xgs'].sum()
        else:
            local_xgf, local_xga = 0, 0
            
        rates[t]['gf_per_game'] = gf / team_games
        rates[t]['ga_per_game'] = ga / team_games
        rates[t]['xgf_per_game'] = xgf / team_games
        rates[t]['xga_per_game'] = xga / team_games
        rates[t]['local_xgf_per_game'] = local_xgf / team_games
        rates[t]['local_xga_per_game'] = local_xga / team_games
        rates[t]['mp_xgf_per_game'] = mp_xgf / team_games
        rates[t]['mp_xga_per_game'] = mp_xga / team_games
        rates[t]['games'] = team_games
        
    return rates

# ---------------------------------------------------------------------------
# Game-State-Aware Per-60 rate extraction
# ---------------------------------------------------------------------------
# League-average time-on-ice per game (seconds) for each game state.
# These are robust league-wide averages that don't change much season to season.
AVG_5V5_TIME_SEC = 48.0 * 60   # ~48 minutes of 5v5 per game
AVG_PP_TIME_SEC  = 6.0 * 60    # ~6 minutes of PP (5v4) per team per game
AVG_PK_TIME_SEC  = 6.0 * 60    # ~6 minutes of PK (4v5) per team per game

def _team_shot_mask(df, team, role='for'):
    """Build a boolean mask for shots FOR or AGAINST a team."""
    if role == 'for':
        return (
            ((df['home_abb'] == team) & (df['team_id'] == df['home_id'])) |
            ((df['away_abb'] == team) & (df['team_id'] == df['away_id']))
        )
    else:  # against
        return (
            ((df['home_abb'] == team) & (df['team_id'] != df['home_id'])) |
            ((df['away_abb'] == team) & (df['team_id'] != df['away_id']))
        )

def extract_rates_per60(train_df):
    """
    Compute per-60 rates broken down by game state (5v5, 5v4, 4v5).
    
    Uses shot-count as a proxy for time-on-ice (shots/time scales linearly
    across teams), then normalises to per-60 using the total shot-count and
    league-average TOI for each state.
    
    Returns dict[team] -> {
        '5v5':  {'gf60', 'ga60', 'xgf60', 'xga60'},
        'pp':   {'gf60', 'xgf60'},   # team's 5v4 offense
        'pk':   {'ga60', 'xga60'},   # team's 4v5 defense
        'games': int
    }
    """
    df = train_df.copy()
    teams = pd.concat([df['home_abb'], df['away_abb']]).unique()
    
    # Pre-split by game state
    df_5v5 = df[df['game_state'] == '5v5']
    df_5v4 = df[df['game_state'] == '5v4']
    df_4v5 = df[df['game_state'] == '4v5']
    
    rates = {}
    
    for t in teams:
        team_games = df[(df['home_abb'] == t) | (df['away_abb'] == t)]['game_id'].nunique()
        if team_games == 0:
            continue
            
        team_hours = team_games  # will divide by team_games later to get per-game, then scale
        
        # --- Helper: count goals and sum xG in a state-filtered df ---
        def _state_stats(state_df, team, role):
            mask = _team_shot_mask(state_df, team, role)
            goals = int((state_df[mask]['event'].str.lower() == 'goal').sum())
            xg    = float(state_df[mask]['xgs'].sum()) if 'xgs' in state_df.columns else 0.0
            mp_xg = float(state_df[mask]['mp_xG'].sum()) if 'mp_xG' in state_df.columns else 0.0
            return goals, xg, mp_xg
        
        # --- 5v5 ---
        gf_5v5, xgf_5v5, _ = _state_stats(df_5v5, t, 'for')
        ga_5v5, xga_5v5, _ = _state_stats(df_5v5, t, 'against')
        
        local_xgf_5v5 = float(df_5v5[_team_shot_mask(df_5v5, t, 'for')]['local_xgs'].sum()) if 'local_xgs' in df_5v5.columns else 0.0
        local_xga_5v5 = float(df_5v5[_team_shot_mask(df_5v5, t, 'against')]['local_xgs'].sum()) if 'local_xgs' in df_5v5.columns else 0.0
        
        # Per-60 = (total / (team_games * avg_state_minutes / 60))
        hours_5v5 = team_games * AVG_5V5_TIME_SEC / 3600
        
        # --- PP (team on the power play) ---
        # If team is Home → PP events are in 5v4.  If Away → PP events are in 4v5.
        # We handle both by looking at "for" shots in each state-df where the team
        # is on the advantageous side.
        # Home team's PP = game_state 5v4;  Away team's PP = game_state 4v5.
        # So: team's PP shots-for = shots by team in 5v4 (when home) + shots by team in 4v5 (when away)
        
        mask_pp_for = (
            ((df_5v4['home_abb'] == t) & (df_5v4['team_id'] == df_5v4['home_id'])) |
            ((df_4v5['away_abb'] == t) & (df_4v5['team_id'] == df_4v5['away_id']))
        )
        # We need to combine both DataFrames for the mask
        df_pp = pd.concat([df_5v4, df_4v5])
        mask_pp_for = (
            ((df_pp['home_abb'] == t) & (df_pp['team_id'] == df_pp['home_id']) & (df_pp['game_state'] == '5v4')) |
            ((df_pp['away_abb'] == t) & (df_pp['team_id'] == df_pp['away_id']) & (df_pp['game_state'] == '4v5'))
        )
        gf_pp = int((df_pp[mask_pp_for]['event'].str.lower() == 'goal').sum())
        xgf_pp = float(df_pp[mask_pp_for]['xgs'].sum()) if 'xgs' in df_pp.columns else 0.0
        local_xgf_pp = float(df_pp[mask_pp_for]['local_xgs'].sum()) if 'local_xgs' in df_pp.columns else 0.0
        
        hours_pp = team_games * AVG_PP_TIME_SEC / 3600
        
        # --- PK (team shorthanded — defending) ---
        # Home team's PK = game_state 4v5 (opponent is on PP);  Away team's PK = game_state 5v4.
        # PK defense = shots AGAINST the team while shorthanded.
        mask_pk_ag = (
            ((df_pp['home_abb'] == t) & (df_pp['team_id'] != df_pp['home_id']) & (df_pp['game_state'] == '4v5')) |
            ((df_pp['away_abb'] == t) & (df_pp['team_id'] != df_pp['away_id']) & (df_pp['game_state'] == '5v4'))
        )
        ga_pk = int((df_pp[mask_pk_ag]['event'].str.lower() == 'goal').sum())
        xga_pk = float(df_pp[mask_pk_ag]['xgs'].sum()) if 'xgs' in df_pp.columns else 0.0
        local_xga_pk = float(df_pp[mask_pk_ag]['local_xgs'].sum()) if 'local_xgs' in df_pp.columns else 0.0
        
        hours_pk = team_games * AVG_PK_TIME_SEC / 3600
        
        rates[t] = {
            '5v5': {
                'gf60':  gf_5v5  / hours_5v5 if hours_5v5 > 0 else 0.0,
                'ga60':  ga_5v5  / hours_5v5 if hours_5v5 > 0 else 0.0,
                'xgf60': xgf_5v5 / hours_5v5 if hours_5v5 > 0 else 0.0,
                'xga60': xga_5v5 / hours_5v5 if hours_5v5 > 0 else 0.0,
                'local_xgf60': local_xgf_5v5 / hours_5v5 if hours_5v5 > 0 else 0.0,
                'local_xga60': local_xga_5v5 / hours_5v5 if hours_5v5 > 0 else 0.0,
            },
            'pp': {
                'gf60':  gf_pp  / hours_pp if hours_pp > 0 else 0.0,
                'xgf60': xgf_pp / hours_pp if hours_pp > 0 else 0.0,
                'local_xgf60': local_xgf_pp / hours_pp if hours_pp > 0 else 0.0,
            },
            'pk': {
                'ga60':  ga_pk  / hours_pk if hours_pk > 0 else 0.0,
                'xga60': xga_pk / hours_pk if hours_pk > 0 else 0.0,
                'local_xga60': local_xga_pk / hours_pk if hours_pk > 0 else 0.0,
            },
            'games': team_games
        }
    
    return rates


def _league_avg_per60(rates_per60, key_path):
    """Compute the league average of a nested per-60 rate across all teams."""
    vals = []
    for t, r in rates_per60.items():
        state, metric = key_path
        if state in r and metric in r[state]:
            vals.append(r[state][metric])
    return np.mean(vals) if vals else 1.0


def predict_matchup_per60(home, away, rates60, xtg_rates, league60):
    """
    Predict expected goals for home and away using game-state-aware per-60 rates.
    
    For each state (5v5, PP, PK), we compute:
        matchup_rate = (off_rate * def_rate) / league_avg_rate
        matchup_xg   = matchup_rate * (avg_state_time / 3600)
    
    Then sum across states for total expected goals.
    """
    def safe_div(n, d, default):
        return (n / d) if d > 0 else default
    
    # Fallback if team missing
    if home not in rates60 or away not in rates60:
        fallback = 3.0
        return {
            'goals': (fallback, fallback),
            'xg': (fallback, fallback),
            'xtg': (fallback, fallback),
            'mp_xg': (fallback, fallback),
        }
    
    h = rates60[home]
    a = rates60[away]
    
    # --- 5v5 ---
    h_5v5_g  = safe_div(h['5v5']['gf60']  * a['5v5']['ga60'],  league60['5v5_g'],  league60['5v5_g'])  * (AVG_5V5_TIME_SEC / 3600)
    a_5v5_g  = safe_div(a['5v5']['gf60']  * h['5v5']['ga60'],  league60['5v5_g'],  league60['5v5_g'])  * (AVG_5V5_TIME_SEC / 3600)
    h_5v5_xg = safe_div(h['5v5']['xgf60'] * a['5v5']['xga60'], league60['5v5_xg'], league60['5v5_xg']) * (AVG_5V5_TIME_SEC / 3600)
    a_5v5_xg = safe_div(a['5v5']['xgf60'] * h['5v5']['xga60'], league60['5v5_xg'], league60['5v5_xg']) * (AVG_5V5_TIME_SEC / 3600)
    h_5v5_lxg = safe_div(h['5v5']['local_xgf60'] * a['5v5']['local_xga60'], league60['5v5_lxg'], league60['5v5_lxg']) * (AVG_5V5_TIME_SEC / 3600) if 'local_xgf60' in h['5v5'] else h_5v5_xg
    a_5v5_lxg = safe_div(a['5v5']['local_xgf60'] * h['5v5']['local_xga60'], league60['5v5_lxg'], league60['5v5_lxg']) * (AVG_5V5_TIME_SEC / 3600) if 'local_xgf60' in a['5v5'] else a_5v5_xg
    
    # --- PP/PK ---
    # Home PP (home offense on PP vs away defense on PK)
    h_pp_g  = safe_div(h['pp']['gf60']  * a['pk']['ga60'],  league60['pp_g'],  league60['pp_g'])  * (AVG_PP_TIME_SEC / 3600)
    a_pk_g  = safe_div(a['pp']['gf60']  * h['pk']['ga60'],  league60['pp_g'],  league60['pp_g'])  * (AVG_PP_TIME_SEC / 3600)
    h_pp_xg = safe_div(h['pp']['xgf60'] * a['pk']['xga60'], league60['pp_xg'], league60['pp_xg']) * (AVG_PP_TIME_SEC / 3600)
    a_pk_xg = safe_div(a['pp']['xgf60'] * h['pk']['xga60'], league60['pp_xg'], league60['pp_xg']) * (AVG_PP_TIME_SEC / 3600)
    h_pp_lxg = safe_div(h['pp']['local_xgf60'] * a['pk']['local_xga60'], league60['pp_lxg'], league60['pp_lxg']) * (AVG_PP_TIME_SEC / 3600) if 'local_xgf60' in h['pp'] else h_pp_xg
    a_pk_lxg = safe_div(a['pp']['local_xgf60'] * h['pk']['local_xga60'], league60['pp_lxg'], league60['pp_lxg']) * (AVG_PP_TIME_SEC / 3600) if 'local_xgf60' in a['pp'] else a_pk_xg
    
    # Totals
    h_g_exp  = h_5v5_g  + h_pp_g
    a_g_exp  = a_5v5_g  + a_pk_g
    h_xg_exp = h_5v5_xg + h_pp_xg
    a_xg_exp = a_5v5_xg + a_pk_xg
    h_lxg_exp = h_5v5_lxg + h_pp_lxg
    a_lxg_exp = a_5v5_lxg + a_pk_lxg
    
    # xtG (uses the same per-game approach as before — already game-state aware internally)
    lg_xtg = league60.get('xtgf_per_game', 3.0)
    if home in xtg_rates and away in xtg_rates:
        h_xtg_exp = safe_div(xtg_rates[home]['xtgf_per_game'] * xtg_rates[away]['xtga_per_game'], lg_xtg, lg_xtg)
        a_xtg_exp = safe_div(xtg_rates[away]['xtgf_per_game'] * xtg_rates[home]['xtga_per_game'], lg_xtg, lg_xtg)
    else:
        h_xtg_exp = lg_xtg
        a_xtg_exp = lg_xtg
    
    # MP xG — not available per-60 in this path, just pass 0
    
    # Local xtG
    lg_lxtg = league60.get('local_xtgf_per_game', 3.0)
    if 'local_xtgf_per_game' in xtg_rates.get(home, {}):
        h_lxtg_exp = safe_div(xtg_rates[home]['local_xtgf_per_game'] * xtg_rates[away]['local_xtga_per_game'], lg_lxtg, lg_lxtg)
        a_lxtg_exp = safe_div(xtg_rates[away]['local_xtgf_per_game'] * xtg_rates[home]['local_xtga_per_game'], lg_lxtg, lg_lxtg)
    else:
        h_lxtg_exp = lg_lxtg
        a_lxtg_exp = lg_lxtg

    return {
        'goals': (h_g_exp, a_g_exp),
        'xg': (h_xg_exp, a_xg_exp),
        'xtg': (h_xtg_exp, a_xtg_exp),
        'mp_xg': (0.0, 0.0),  # not computed in per-60 variant
        'local_xg': (h_lxg_exp, a_lxg_exp),
        'local_xtg': (h_lxtg_exp, a_lxtg_exp),
    }


def train_mixed_effects(train_df):
    """Train the xtG model on the partition to get team intercepts/rates."""
    logger.info("Training Mixed Effects Model on sample partition...")
    
    # Ensure is_goal exists
    if 'is_goal' not in train_df.columns:
        train_df['is_goal'] = (train_df['event'].str.lower() == 'goal').astype(int)
        
    mixed = mixed_effects.GameMixedEffectsXG(
        base_model_path="analysis/xgs/xg_model_nested_tensor.joblib",
        feature_set=[], 
        use_tensor_splines=True, 
        component_model_type='intercept_and_features',
        l2_reg=1.0 
    )
    
    mixed.fit(train_df)
    # To properly extract an SOS-independent rate, we must neutralize the opponent
    df_off = train_df.copy()
    is_home_shot = df_off['team_id'] == df_off['home_id']
    df_off['off_team_name'] = np.where(is_home_shot, df_off['home_abb'], df_off['away_abb'])
    df_off['def_team_name'] = 'Average' # Neutralize defense!
    
    df_def = train_df.copy()
    df_def['off_team_name'] = 'Average' # Neutralize offense!
    df_def['def_team_name'] = np.where(is_home_shot, df_def['away_abb'], df_def['home_abb'])
    
    df_off['xtgf_gen'] = mixed.predict_proba(df_off)[:, 1]
    df_def['xtga_all'] = mixed.predict_proba(df_def)[:, 1]
    
    teams = pd.concat([train_df['home_abb'], train_df['away_abb']]).unique()
    xtg_rates = {}
    
    for t in teams:
        team_games = train_df[(train_df['home_abb'] == t) | (train_df['away_abb'] == t)]['game_id'].nunique()
        if team_games == 0: continue
        
        # xtgf: Sum of generated expected goals (against average defense)
        xtgf = df_off[((df_off['home_abb'] == t) & (df_off['team_id'] == df_off['home_id'])) | ((df_off['away_abb'] == t) & (df_off['team_id'] == df_off['away_id']))]['xtgf_gen'].sum()
        
        # xtga: Sum of allowed expected goals (against average offense)
        xtga = df_def[((df_def['home_abb'] == t) & (df_def['team_id'] != df_def['home_id'])) | ((df_def['away_abb'] == t) & (df_def['team_id'] != df_def['away_id']))]['xtga_all'].sum()
        
        xtg_rates[t] = {
            'xtgf_per_game': xtgf / team_games,
            'xtga_per_game': xtga / team_games,
        }
        
    return xtg_rates

def train_local_models(train_df):
    """Train the pure local nested model and local mixed effects model on the train_df slice."""
    logger.info("Training Local Nested Model...")
    
    # 1. Train Nested GLM
    feature_list = feature_util.get_features('all_inclusive')
    local_xg_clf = fit_glm_nested.NestedGLM(
        features=feature_list,
        poly_degree=2,
        use_splines=True,
        enable_marginalization=True
    )
    
    local_xg_clf.fit(train_df)
    train_df['local_xgs'] = local_xg_clf.predict_proba(train_df)[:, 1]
    
    # 2. Train Local Mixed Effects
    logger.info("Training Local Mixed Effects Model (based on Local Nested)...")
    if 'is_goal' not in train_df.columns:
        train_df['is_goal'] = (train_df['event'].str.lower() == 'goal').astype(int)
        
    local_mixed = mixed_effects.GameMixedEffectsXG(
        base_model=local_xg_clf,
        feature_set=[], 
        use_tensor_splines=True, 
        component_model_type='intercept_and_features',
        l2_reg=1.0 
    )
    
    local_mixed.fit(train_df)
    
    df_off = train_df.copy()
    is_home_shot = df_off['team_id'] == df_off['home_id']
    df_off['off_team_name'] = np.where(is_home_shot, df_off['home_abb'], df_off['away_abb'])
    df_off['def_team_name'] = 'Average' # Neutralize defense!
    
    df_def = train_df.copy()
    df_def['off_team_name'] = 'Average' # Neutralize offense!
    df_def['def_team_name'] = np.where(is_home_shot, df_def['away_abb'], df_def['home_abb'])
    
    df_off['local_xtgf_gen'] = local_mixed.predict_proba(df_off)[:, 1]
    df_def['local_xtga_all'] = local_mixed.predict_proba(df_def)[:, 1]
    
    teams = pd.concat([train_df['home_abb'], train_df['away_abb']]).unique()
    local_xtg_rates = {}
    
    for t in teams:
        team_games = train_df[(train_df['home_abb'] == t) | (train_df['away_abb'] == t)]['game_id'].nunique()
        if team_games == 0: continue
        
        local_xtgf = df_off[((df_off['home_abb'] == t) & (df_off['team_id'] == df_off['home_id'])) | ((df_off['away_abb'] == t) & (df_off['team_id'] == df_off['away_id']))]['local_xtgf_gen'].sum()
        local_xtga = df_def[((df_def['home_abb'] == t) & (df_def['team_id'] != df_def['home_id'])) | ((df_def['away_abb'] == t) & (df_def['team_id'] != df_def['away_id']))]['local_xtga_all'].sum()
        
        local_xtg_rates[t] = {
            'local_xtgf_per_game': local_xtgf / team_games,
            'local_xtga_per_game': local_xtga / team_games,
        }
        
    return local_xtg_rates, train_df

def predict_matchup(home, away, rates, xtg_rates, league_avgs):
    """Predict the expected goals for home and away using the three methodologies."""
    
    def safe_div(n, d, default):
        return (n / d) if d > 0 else default
        
    # 1. Actual Goals
    if home in rates and away in rates:
        h_g_exp = safe_div(rates[home]['gf_per_game'] * rates[away]['ga_per_game'], league_avgs['gf_per_game'], league_avgs['gf_per_game'])
        a_g_exp = safe_div(rates[away]['gf_per_game'] * rates[home]['ga_per_game'], league_avgs['gf_per_game'], league_avgs['gf_per_game'])
    else:
        h_g_exp = league_avgs['gf_per_game']
        a_g_exp = league_avgs['gf_per_game']
        
    # 2. Standard xG
    if home in rates and away in rates:
        h_xg_exp = safe_div(rates[home]['xgf_per_game'] * rates[away]['xga_per_game'], league_avgs['xgf_per_game'], league_avgs['xgf_per_game'])
        a_xg_exp = safe_div(rates[away]['xgf_per_game'] * rates[home]['xga_per_game'], league_avgs['xgf_per_game'], league_avgs['xgf_per_game'])
    else:
        h_xg_exp = league_avgs['xgf_per_game']
        a_xg_exp = league_avgs['xgf_per_game']
        
    # 3. Mixed Effects xtG
    if home in xtg_rates and away in xtg_rates:
        h_xtg_exp = safe_div(xtg_rates[home]['xtgf_per_game'] * xtg_rates[away]['xtga_per_game'], league_avgs['xtgf_per_game'], league_avgs['xtgf_per_game'])
        a_xtg_exp = safe_div(xtg_rates[away]['xtgf_per_game'] * xtg_rates[home]['xtga_per_game'], league_avgs['xtgf_per_game'], league_avgs['xtgf_per_game'])
    else:
        h_xtg_exp = league_avgs['xtgf_per_game']
        a_xtg_exp = league_avgs['xtgf_per_game']
        
    # 4. MoneyPuck xG
    if home in rates and away in rates:
        h_mp_exp = safe_div(rates[home]['mp_xgf_per_game'] * rates[away]['mp_xga_per_game'], league_avgs['mp_xgf_per_game'], league_avgs['mp_xgf_per_game'])
        a_mp_exp = safe_div(rates[away]['mp_xgf_per_game'] * rates[home]['mp_xga_per_game'], league_avgs['mp_xgf_per_game'], league_avgs['mp_xgf_per_game'])
    else:
        h_mp_exp = league_avgs['mp_xgf_per_game']
        a_mp_exp = league_avgs['mp_xgf_per_game']
        
    # 5. Local xG
    if home in rates and away in rates:
        h_lxg_exp = safe_div(rates[home]['local_xgf_per_game'] * rates[away]['local_xga_per_game'], league_avgs['local_xgf_per_game'], league_avgs['local_xgf_per_game'])
        a_lxg_exp = safe_div(rates[away]['local_xgf_per_game'] * rates[home]['local_xga_per_game'], league_avgs['local_xgf_per_game'], league_avgs['local_xgf_per_game'])
    else:
        h_lxg_exp = league_avgs.get('local_xgf_per_game', 3.0)
        a_lxg_exp = league_avgs.get('local_xgf_per_game', 3.0)
        
    # 6. Local xtG
    if home in xtg_rates and away in xtg_rates and 'local_xtgf_per_game' in xtg_rates[home]:
        h_lxtg_exp = safe_div(xtg_rates[home]['local_xtgf_per_game'] * xtg_rates[away]['local_xtga_per_game'], league_avgs['local_xtgf_per_game'], league_avgs['local_xtgf_per_game'])
        a_lxtg_exp = safe_div(xtg_rates[away]['local_xtgf_per_game'] * xtg_rates[home]['local_xtga_per_game'], league_avgs['local_xtgf_per_game'], league_avgs['local_xtgf_per_game'])
    else:
        h_lxtg_exp = league_avgs.get('local_xtgf_per_game', 3.0)
        a_lxtg_exp = league_avgs.get('local_xtgf_per_game', 3.0)
        
    return {
        'goals': (h_g_exp, a_g_exp),
        'xg': (h_xg_exp, a_xg_exp),
        'xtg': (h_xtg_exp, a_xtg_exp),
        'mp_xg': (h_mp_exp, a_mp_exp),
        'local_xg': (h_lxg_exp, a_lxg_exp),
        'local_xtg': (h_lxtg_exp, a_lxtg_exp)
    }

def calculate_win_prob(h_exp, a_exp):
    """Use Poisson to get Home Win, Away Win, Tie probabilities."""
    # Cap expected goals to prevent numerical issues
    h_exp = min(max(h_exp, 0.1), 10.0)
    a_exp = min(max(a_exp, 0.1), 10.0)
    
    max_goals = 15
    h_dist = poisson.pmf(np.arange(max_goals), h_exp)
    a_dist = poisson.pmf(np.arange(max_goals), a_exp)
    
    # Outer product for joint probability matrix
    prob_matrix = np.outer(h_dist, a_dist)
    
    hw = np.tril(prob_matrix, -1).sum()
    aw = np.triu(prob_matrix, 1).sum()
    tie = np.trace(prob_matrix)
    
    # Normalize just in case 15 goals truncates too much
    total = hw + aw + tie
    if total > 0:
        return hw/total, aw/total, tie/total
    return 0.33, 0.33, 0.33

def get_actual_result(home_goals, away_goals):
    """Return 1 for home win, 0 for away win. (Assuming Ties split 0.5 for log loss)."""
    if home_goals > away_goals: return 1.0, 0.0, 0.0
    if away_goals > home_goals: return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=1000, ci=95, rng_seed=42):
    """
    Bootstrap a scalar metric to get mean + confidence interval.
    
    Args:
        y_true: array of actual outcomes
        y_pred: array of predicted probabilities
        metric_fn: callable(y_true, y_pred) -> float
        n_boot: number of bootstrap iterations
        ci: confidence interval width (e.g. 95)
    
    Returns:
        (mean, lo, hi) where lo/hi are the CI bounds
    """
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = metric_fn(y_true[idx], y_pred[idx])
    
    # Handle NaNs that might arise from spearman on small perfectly matched/unmatched subsets
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return np.nan, np.nan, np.nan
        
    alpha = (100 - ci) / 2
    return float(np.mean(boots)), float(np.percentile(boots, alpha)), float(np.percentile(boots, 100 - alpha))


def _brier(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))

def _accuracy(y_true, y_pred):
    return float(np.mean((y_pred > 0.5) == (y_true > 0.5)))

def _spearman(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return float(r)



def main():
    parser = argparse.ArgumentParser(description="Evaluate Predictive Power")
    parser.add_argument('--seasons', type=str, default='all_82', help="Comma-separated list of seasons, or 'all_82' for full seasons")
    parser.add_argument('--games-range', type=str, default='10,40,10', help='Start,End,Step for N games')
    parser.add_argument('--state', type=str, default='all', help='5v5 or all')
    parser.add_argument('--score-effect-filter', action='store_true', help='Filter to Close games (diff between -1, 0, 1)')
    parser.add_argument('--per60', action='store_true', help='Use game-state-aware per-60 rates (5v5, PP, PK)')
    parser.add_argument('--n-boot', type=int, default=1000, help='Number of bootstrap iterations (0 to disable)')
    args = parser.parse_args()
    
    if args.seasons == 'all_82':
        # Exclude shortened 20192020 and 20202021, and current 20252026.
        # Add historical seasons as they become available.
        seasons = ['20142015', '20152016', '20162017', '20172018', '20182019', '20212022', '20222023', '20232024', '20242025']
    else:
        seasons = [s.strip() for s in args.seasons.split(',')]
        
    g_start, g_end, g_step = map(int, args.games_range.split(','))
    n_games_list = list(range(g_start, g_end + 1, g_step))
    
    all_seasons_results = []
    suffix = f"{args.state}_per60" if args.per60 else args.state
    
    for season in seasons:
        logger.info(f"\n=======================================================")
        logger.info(f"--- Starting Predictive Evaluation for {season} ---")
        logger.info(f"=======================================================")
        logger.info(f"Testing N Games: {n_games_list}")
        
        # 1. Load Data
        try:
            csv_path = analyze.locate_season_csv(season)
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to load data for {season}: {e}")
            continue
        
        # Pre-process for models (run centralized data pipeline to impute blocked coords, etc)
        from puck import data_pipeline
        df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=True, apply_arena_adjustments=True, apply_bio_enrichment=True, apply_filtering=True
        )
        
        # Ensure standard xG is present
        df, _, _ = analyze._predict_xgs(df)
        
        # Filter state if needed
        if args.state != 'all':
            df = df[df['game_state'] == args.state].copy()
            
        if args.score_effect_filter:
            # We need the current goal differential. The dataset usually provides home_score / away_score prior to the event.
            if 'home_score' in df.columns and 'away_score' in df.columns:
                diff = df['home_score'] - df['away_score']
                df = df[diff.abs() <= 1].copy()
                logger.info(f"Applied Score Effect Filter. Remaining events: {len(df)}")
            else:
                logger.warning("Could not find 'home_score' or 'away_score'. Cannot apply score effect filter.")
            
        sched_df = process_schedule_from_events(df)
        team_schedules = get_team_game_counts(sched_df)
        
        # Calculate EOS Standings
        eos_standings = calculate_standings(sched_df)
        eos_ranks = pd.Series({t: stats['pts_pct'] for t, stats in eos_standings.items()}).rank(ascending=False)
        
        results_list = []
        
        for n in n_games_list:
            logger.info(f"\n--- Evaluating Models at N={n} Games ---")
            train_df, test_df = split_data(df, team_schedules, n)
            
            logger.info(f"Train Events: {len(train_df)} | Test Events: {len(test_df)}")
            
            if len(test_df) == 0:
                logger.warning("No test games found (season might be too short for this N). Skipping.")
                continue
                
            # Extract Rates
            xtg_rates = train_mixed_effects(train_df)
            local_xtg_rates, train_df = train_local_models(train_df)
            
            # Merge local_xtg_rates into xtg_rates for easy passing
            for t in local_xtg_rates:
                if t in xtg_rates:
                    xtg_rates[t].update(local_xtg_rates[t])
                else:
                    xtg_rates[t] = local_xtg_rates[t]
            
            rates60, rates = None, None
            if args.per60:
                rates60 = extract_rates_per60(train_df)
                teams_w_data = [t for t in rates60.keys() if rates60[t]['games'] > 0]
                if not teams_w_data: continue
                league_avgs = {
                    '5v5_g':  _league_avg_per60(rates60, ('5v5', 'gf60')),
                    '5v5_xg': _league_avg_per60(rates60, ('5v5', 'xgf60')),
                    '5v5_lxg': _league_avg_per60(rates60, ('5v5', 'local_xgf60')),
                    'pp_g':   _league_avg_per60(rates60, ('pp', 'gf60')),
                    'pp_xg':  _league_avg_per60(rates60, ('pp', 'xgf60')),
                    'pp_lxg': _league_avg_per60(rates60, ('pp', 'local_xgf60')),
                    'xtgf_per_game': np.mean([xtg_rates[t]['xtgf_per_game'] for t in teams_w_data if t in xtg_rates]),
                    'local_xtgf_per_game': np.mean([local_xtg_rates[t]['local_xtgf_per_game'] for t in teams_w_data if t in local_xtg_rates])
                }
            else:
                rates = extract_rates(train_df)
                teams_w_data = [t for t in rates.keys() if rates[t]['games'] > 0]
                if not teams_w_data: continue
                league_avgs = {
                    'gf_per_game': np.mean([rates[t]['gf_per_game'] for t in teams_w_data]),
                    'xgf_per_game': np.mean([rates[t]['xgf_per_game'] for t in teams_w_data]),
                    'mp_xgf_per_game': np.mean([rates[t]['mp_xgf_per_game'] for t in teams_w_data]),
                    'local_xgf_per_game': np.mean([rates[t]['local_xgf_per_game'] for t in teams_w_data]),
                    'xtgf_per_game': np.mean([xtg_rates[t]['xtgf_per_game'] for t in teams_w_data if t in xtg_rates]),
                    'local_xtgf_per_game': np.mean([local_xtg_rates[t]['local_xtgf_per_game'] for t in teams_w_data if t in local_xtg_rates])
                }
            
            # Now predict games in the test set
            test_sched = process_schedule_from_events(test_df)
            
            y_actual_home = []
            p_goals_home = []
            p_xg_home = []
            p_xtg_home = []
            p_lxg_home = []
            p_lxtg_home = []
            p_mp_home = []
            
            for _, row in test_sched.iterrows():
                h = row['home_team']
                a = row['away_team']
                
                h_wins, a_wins, ties = get_actual_result(row['home_goals'], row['away_goals'])
                actual_points = h_wins * 1.0 + ties * 0.5
                y_actual_home.append(actual_points)
                
                if args.per60:
                    exps = predict_matchup_per60(h, a, rates60, xtg_rates, league_avgs)
                else:
                    exps = predict_matchup(h, a, rates, xtg_rates, league_avgs)
                
                # Predict Goals Base
                hw_g, aw_g, tie_g = calculate_win_prob(exps['goals'][0], exps['goals'][1])
                p_goals_home.append(hw_g + tie_g * 0.5)
                
                # Predict xG Base
                hw_x, aw_x, tie_x = calculate_win_prob(exps['xg'][0], exps['xg'][1])
                p_xg_home.append(hw_x + tie_x * 0.5)
                
                # Predict xtG Base
                hw_xt, aw_xt, tie_xt = calculate_win_prob(exps['xtg'][0], exps['xtg'][1])
                p_xtg_home.append(hw_xt + tie_xt * 0.5)
                
                # Predict Local xG Base
                hw_lx, aw_lx, tie_lx = calculate_win_prob(exps['local_xg'][0], exps['local_xg'][1])
                p_lxg_home.append(hw_lx + tie_lx * 0.5)
                
                # Predict Local xtG Base
                hw_lxt, aw_lxt, tie_lxt = calculate_win_prob(exps['local_xtg'][0], exps['local_xtg'][1])
                p_lxtg_home.append(hw_lxt + tie_lxt * 0.5)
                
                # Predict MP xG Base
                if 'mp_xg' in exps:
                    hw_mp, aw_mp, tie_mp = calculate_win_prob(exps['mp_xg'][0], exps['mp_xg'][1])
                    p_mp_home.append(hw_mp + tie_mp * 0.5)
                
            # Convert to arrays
            y_act = np.array(y_actual_home)
            p_g   = np.array(p_goals_home)
            p_x   = np.array(p_xg_home)
            p_xt  = np.array(p_xtg_home)
            p_lx  = np.array(p_lxg_home)
            p_lxt = np.array(p_lxtg_home)
            p_mp  = np.array(p_mp_home) if p_mp_home else np.zeros_like(y_act)
            
            n_boot = args.n_boot
            
            if n_boot > 0 and len(y_act) >= 10:
                # Bootstrap Brier
                bg_mean, bg_lo, bg_hi = bootstrap_metric(y_act, p_g,  _brier, n_boot)
                bx_mean, bx_lo, bx_hi = bootstrap_metric(y_act, p_x,  _brier, n_boot)
                bt_mean, bt_lo, bt_hi = bootstrap_metric(y_act, p_xt, _brier, n_boot)
                blx_mean, blx_lo, blx_hi = bootstrap_metric(y_act, p_lx, _brier, n_boot)
                blt_mean, blt_lo, blt_hi = bootstrap_metric(y_act, p_lxt, _brier, n_boot)
                bm_mean, bm_lo, bm_hi = bootstrap_metric(y_act, p_mp, _brier, n_boot) if p_mp_home else (0,0,0)
                
                # Bootstrap Accuracy
                ag_mean, ag_lo, ag_hi = bootstrap_metric(y_act, p_g,  _accuracy, n_boot)
                ax_mean, ax_lo, ax_hi = bootstrap_metric(y_act, p_x,  _accuracy, n_boot)
                at_mean, at_lo, at_hi = bootstrap_metric(y_act, p_xt, _accuracy, n_boot)
                alx_mean, alx_lo, alx_hi = bootstrap_metric(y_act, p_lx, _accuracy, n_boot)
                alt_mean, alt_lo, alt_hi = bootstrap_metric(y_act, p_lxt, _accuracy, n_boot)
                am_mean, am_lo, am_hi = bootstrap_metric(y_act, p_mp, _accuracy, n_boot) if p_mp_home else (0,0,0)
            else:
                # Fallback to point estimates (too few games for bootstrap)
                bg_mean = _brier(y_act, p_g);  bg_lo = bg_hi = bg_mean
                bx_mean = _brier(y_act, p_x);  bx_lo = bx_hi = bx_mean
                bt_mean = _brier(y_act, p_xt); bt_lo = bt_hi = bt_mean
                blx_mean = _brier(y_act, p_lx); blx_lo = blx_hi = blx_mean
                blt_mean = _brier(y_act, p_lxt); blt_lo = blt_hi = blt_mean
                bm_mean = _brier(y_act, p_mp) if p_mp_home else 0.0; bm_lo = bm_hi = bm_mean
                ag_mean = _accuracy(y_act, p_g);  ag_lo = ag_hi = ag_mean
                ax_mean = _accuracy(y_act, p_x);  ax_lo = ax_hi = ax_mean
                at_mean = _accuracy(y_act, p_xt); at_lo = at_hi = at_mean
                alx_mean = _accuracy(y_act, p_lx); alx_lo = alx_hi = alx_mean
                alt_mean = _accuracy(y_act, p_lxt); alt_lo = alt_hi = alt_mean
                am_mean = _accuracy(y_act, p_mp) if p_mp_home else 0.0; am_lo = am_hi = am_mean
            
            
            # Calculate ROS Standings for this N
            # Test schedule contains all games AFTER N for both teams (strictly ROS for both)
            ros_standings = calculate_standings(test_sched)
            # Only rank teams that have ROS games
            ros_pts_pct = {t: stats['pts_pct'] for t, stats in ros_standings.items() if stats['games'] > 0}
            ros_ranks = pd.Series(ros_pts_pct).rank(ascending=False)
            
            # Calculate Model Net Strengths
            model_ranks = {}
            for metric, extract_fn in [
                ('Goals', lambda t: rates[t]['gf_per_game'] - rates[t]['ga_per_game'] if not args.per60 else rates60[t]['5v5']['gf60'] - rates60[t]['5v5']['ga60'] + rates60[t]['pp']['gf60'] - rates60[t]['pk']['ga60']),
                ('xG', lambda t: rates[t]['xgf_per_game'] - rates[t]['xga_per_game'] if not args.per60 else rates60[t]['5v5']['xgf60'] - rates60[t]['5v5']['xga60'] + rates60[t]['pp']['xgf60'] - rates60[t]['pk']['xga60']),
                ('Local_xG', lambda t: rates[t]['local_xgf_per_game'] - rates[t]['local_xga_per_game'] if not args.per60 else rates60[t]['5v5']['local_xgf60'] - rates60[t]['5v5']['local_xga60'] + rates60[t]['pp']['local_xgf60'] - rates60[t]['pk']['local_xga60']),
                ('xtG', lambda t: xtg_rates[t]['xtgf_per_game'] - xtg_rates[t]['xtga_per_game']),
                ('Local_xtG', lambda t: xtg_rates[t]['local_xtgf_per_game'] - xtg_rates[t]['local_xtga_per_game']),
                ('MP_xG', lambda t: rates[t]['mp_xgf_per_game'] - rates[t]['mp_xga_per_game'] if not args.per60 else 0)
            ]:
                if metric == 'MP_xG' and args.per60: continue
                metric_vals = {}
                for t in teams_w_data:
                    try:
                        metric_vals[t] = extract_fn(t)
                    except KeyError:
                        pass
                model_ranks[metric] = pd.Series(metric_vals).rank(ascending=False)
                
            # Calculate Spearman Correlations
            spearman_eos = {}
            spearman_ros = {}
            for metric, ranks in model_ranks.items():
                # MATCH EOS
                common_eos = ranks.index.intersection(eos_ranks.index)
                if len(common_eos) >= 5: # Need a minimum number of teams to rank
                    y_true = eos_ranks[common_eos].values
                    y_pred = ranks[common_eos].values
                    if n_boot > 0:
                        r_mean, r_lo, r_hi = bootstrap_metric(y_true, y_pred, _spearman, n_boot)
                    else:
                        r_mean = _spearman(y_true, y_pred)
                        r_lo = r_hi = r_mean
                        
                    spearman_eos[f'{metric}_Spearman_EOS'] = r_mean
                    spearman_eos[f'{metric}_Spearman_EOS_lo'] = r_lo
                    spearman_eos[f'{metric}_Spearman_EOS_hi'] = r_hi
                else:
                    spearman_eos[f'{metric}_Spearman_EOS'] = np.nan
                    spearman_eos[f'{metric}_Spearman_EOS_lo'] = np.nan
                    spearman_eos[f'{metric}_Spearman_EOS_hi'] = np.nan
                    
                # MATCH ROS
                common_ros = ranks.index.intersection(ros_ranks.index)
                if len(common_ros) >= 5:
                    y_true = ros_ranks[common_ros].values
                    y_pred = ranks[common_ros].values
                    if n_boot > 0:
                        r_mean, r_lo, r_hi = bootstrap_metric(y_true, y_pred, _spearman, n_boot)
                    else:
                        r_mean = _spearman(y_true, y_pred)
                        r_lo = r_hi = r_mean
                        
                    spearman_ros[f'{metric}_Spearman_ROS'] = r_mean
                    spearman_ros[f'{metric}_Spearman_ROS_lo'] = r_lo
                    spearman_ros[f'{metric}_Spearman_ROS_hi'] = r_hi
                else:
                    spearman_ros[f'{metric}_Spearman_ROS'] = np.nan
                    spearman_ros[f'{metric}_Spearman_ROS_lo'] = np.nan
                    spearman_ros[f'{metric}_Spearman_ROS_hi'] = np.nan
                    
            res = {
                'N': n,
                'Test_Games': len(test_sched),
                'Goals_Brier': bg_mean, 'Goals_Brier_lo': bg_lo, 'Goals_Brier_hi': bg_hi,
                'xG_Brier': bx_mean,    'xG_Brier_lo': bx_lo,    'xG_Brier_hi': bx_hi,
                'Local_xG_Brier': blx_mean, 'Local_xG_Brier_lo': blx_lo, 'Local_xG_Brier_hi': blx_hi,
                'xtG_Brier': bt_mean,   'xtG_Brier_lo': bt_lo,   'xtG_Brier_hi': bt_hi,
                'Local_xtG_Brier': blt_mean, 'Local_xtG_Brier_lo': blt_lo, 'Local_xtG_Brier_hi': blt_hi,
                'MP_Brier': bm_mean,    'MP_Brier_lo': bm_lo,    'MP_Brier_hi': bm_hi,
                'Goals_Acc': ag_mean,   'Goals_Acc_lo': ag_lo,    'Goals_Acc_hi': ag_hi,
                'xG_Acc': ax_mean,      'xG_Acc_lo': ax_lo,      'xG_Acc_hi': ax_hi,
                'Local_xG_Acc': alx_mean,   'Local_xG_Acc_lo': alx_lo,    'Local_xG_Acc_hi': alx_hi,
                'xtG_Acc': at_mean,     'xtG_Acc_lo': at_lo,     'xtG_Acc_hi': at_hi,
                'Local_xtG_Acc': alt_mean,  'Local_xtG_Acc_lo': alt_lo,   'Local_xtG_Acc_hi': alt_hi,
                'MP_Acc': am_mean,      'MP_Acc_lo': am_lo,      'MP_Acc_hi': am_hi,
            }
            res.update(spearman_eos)
            res.update(spearman_ros)
            results_list.append(res)
            logger.info(f"Results for N={n}: \n Brier [Goals: {bg_mean:.4f}±{bg_hi-bg_lo:.4f}, xG: {bx_mean:.4f}±{bx_hi-bx_lo:.4f}, "
                        f"LxG: {blx_mean:.4f}±{blx_hi-blx_lo:.4f}, xtG: {bt_mean:.4f}±{bt_hi-bt_lo:.4f}, "
                        f"LxtG: {blt_mean:.4f}±{blt_hi-blt_lo:.4f}, MP: {bm_mean:.4f}±{bm_hi-bm_lo:.4f}]")
            
        df_res = pd.DataFrame(results_list)
        df_res['Season'] = season
        all_seasons_results.append(df_res)
        
        out_path = Path(f"analysis/evaluation/predictive_power_{season}_{suffix}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_path, index=False)
        
        # Generate per-season plots using the Plotting script logic
        plot_predictive_power.generate_plots(out_path, out_path.parent)
        
        # Generate per-season Spearman Plots
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("husl", 6)
        metrics_to_plot = ['Goals', 'xG', 'Local_xG', 'xtG', 'Local_xtG', 'MP_xG']
        
        # Plot Spearman EOS
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, metric in enumerate(metrics_to_plot):
            col = f'{metric}_Spearman_EOS'
            lo_col = f'{metric}_Spearman_EOS_lo'
            hi_col = f'{metric}_Spearman_EOS_hi'
            label = metric.replace('_', ' ')
            
            if col in df_res.columns and not df_res[col].isna().all():
                color = palette[idx]
                sns.lineplot(data=df_res, x='N', y=col, marker='o', label=label, ax=ax, color=color)
                
                # Add confidence interval shading if available
                if lo_col in df_res.columns and hi_col in df_res.columns:
                    ax.fill_between(df_res['N'], df_res[lo_col].astype(float), df_res[hi_col].astype(float), color=color, alpha=0.2)
        
        ax.set_title(f'Rank correlation with End-of-Season Standings ({season})')
        ax.set_xlabel('Number of Training Games (N)')
        ax.set_ylabel('Spearman Correlation Coefficient')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_eos_path = Path(f"analysis/evaluation/spearman_correlation_eos_{season}_{suffix}.png")
        plt.savefig(plot_eos_path)
        plt.close(fig)
        logger.info(f"Saved EOS Spearman plot to {plot_eos_path}")
        
        # Plot Spearman ROS
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, metric in enumerate(metrics_to_plot):
            col = f'{metric}_Spearman_ROS'
            lo_col = f'{metric}_Spearman_ROS_lo'
            hi_col = f'{metric}_Spearman_ROS_hi'
            label = metric.replace('_', ' ')
            
            if col in df_res.columns and not df_res[col].isna().all():
                color = palette[idx]
                sns.lineplot(data=df_res, x='N', y=col, marker='o', label=label, ax=ax, color=color)
                
                # Add confidence interval shading if available
                if lo_col in df_res.columns and hi_col in df_res.columns:
                    ax.fill_between(df_res['N'], df_res[lo_col].astype(float), df_res[hi_col].astype(float), color=color, alpha=0.2)
        
        ax.set_title(f'Rank correlation with Rest-of-Season Standings ({season})')
        ax.set_xlabel('Number of Training Games (N)')
        ax.set_ylabel('Spearman Correlation Coefficient')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_ros_path = Path(f"analysis/evaluation/spearman_correlation_ros_{season}_{suffix}.png")
        plt.savefig(plot_ros_path)
        plt.close(fig)
        logger.info(f"Saved ROS Spearman plot to {plot_ros_path}")

        print("\n--- SEASON RESULTS ---")
        print(df_res.to_string(index=False))

    if not all_seasons_results:
        logger.warning("No seasons were processed. Exiting.")
        return

    # --- AGGREGATE RESULTS ---
    logger.info(f"\n=======================================================")
    logger.info(f"--- Calculating Aggregate Plots over {len(all_seasons_results)} seasons ---")
    logger.info(f"=======================================================")
    
    df_all = pd.concat(all_seasons_results, ignore_index=True)
    
    # We want to group by N to get the mean metrics.
    # Note: we drop "Season" which is non-numeric, and "Test_Games" we should probably sum or keep as mean.
    agg_funcs = {col: 'mean' for col in df_all.columns if col not in ['N', 'Season']}
    agg_funcs['Test_Games'] = 'sum'
    
    df_agg = df_all.groupby('N').agg(agg_funcs).reset_index()
    
    out_agg_path = Path(f"analysis/evaluation/predictive_power_aggregate_{suffix}.csv")
    df_agg.to_csv(out_agg_path, index=False)
    
    # Generate aggregate plots for Accuracy/Brier using standard plotter
    plot_predictive_power.generate_plots(out_agg_path, out_agg_path.parent)
    
    # Generate Aggregate Spearman plots
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ['Goals', 'xG', 'Local_xG', 'xtG', 'Local_xtG', 'MP_xG']
    palette = sns.color_palette("husl", 6)
    
    # Aggregate EOS Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, metric in enumerate(metrics_to_plot):
        col = f'{metric}_Spearman_EOS'
        lo_col = f'{metric}_Spearman_EOS_lo'
        hi_col = f'{metric}_Spearman_EOS_hi'
        label = metric.replace('_', ' ')
        
        if col in df_agg.columns and not df_agg[col].isna().all():
            color = palette[idx]
            sns.lineplot(data=df_agg, x='N', y=col, marker='o', label=label, ax=ax, color=color)
            if lo_col in df_agg.columns and hi_col in df_agg.columns:
                ax.fill_between(df_agg['N'], df_agg[lo_col].astype(float), df_agg[hi_col].astype(float), color=color, alpha=0.2)
                
    ax.set_title(f'Rank Correlation with End-of-Season Standings (Aggregated)')
    ax.set_xlabel('Number of Training Games (N)')
    ax.set_ylabel('Spearman Correlation Coefficient')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_agg_eos_path = Path(f"analysis/evaluation/spearman_correlation_eos_aggregate_{suffix}.png")
    plt.savefig(plot_agg_eos_path)
    plt.close(fig)
    logger.info(f"Saved Aggregate EOS Spearman plot to {plot_agg_eos_path}")

    # Aggregate ROS Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, metric in enumerate(metrics_to_plot):
        col = f'{metric}_Spearman_ROS'
        lo_col = f'{metric}_Spearman_ROS_lo'
        hi_col = f'{metric}_Spearman_ROS_hi'
        label = metric.replace('_', ' ')
        
        if col in df_agg.columns and not df_agg[col].isna().all():
            color = palette[idx]
            sns.lineplot(data=df_agg, x='N', y=col, marker='o', label=label, ax=ax, color=color)
            if lo_col in df_agg.columns and hi_col in df_agg.columns:
                ax.fill_between(df_agg['N'], df_agg[lo_col].astype(float), df_agg[hi_col].astype(float), color=color, alpha=0.2)
                
    ax.set_title(f'Rank Correlation with Rest-of-Season Standings (Aggregated)')
    ax.set_xlabel('Number of Training Games (N)')
    ax.set_ylabel('Spearman Correlation Coefficient')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_agg_ros_path = Path(f"analysis/evaluation/spearman_correlation_ros_aggregate_{suffix}.png")
    plt.savefig(plot_agg_ros_path)
    plt.close(fig)
    logger.info(f"Saved Aggregate ROS Spearman plot to {plot_agg_ros_path}")

    print("\n--- AGGREGATE RESULTS ---")
    print(df_agg.to_string(index=False))
    print(f"\nSaved Aggregate CSV to {out_agg_path}")

if __name__ == "__main__":
    main()
