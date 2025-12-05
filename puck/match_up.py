import os
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from . import timing
from . import analyze
from . import nhl_api

class GamePredictor:
    def __init__(self, season='20252026', weight_decay=0.05):
        """
        Initialize the GamePredictor.
        
        Args:
            season (str): Season string (e.g., '20252026').
            weight_decay (float): Decay factor for recency weighting. 
                                  0.0 = Equal weighting. 
                                  Higher = More weight to recent games.
        """
        self.season = season
        self.weight_decay = weight_decay
        self.df_season = None
        self.team_stats = {} # Cache for team stats
        
        # Load data immediately
        self.load_data()

    def load_data(self):
        """Loads season data and prepares it for analysis."""
        print(f"Loading data for season {self.season}...")
        self.df_season = timing.load_season_df(self.season)
        
        if self.df_season is None or self.df_season.empty:
            raise ValueError("No season data found.")
            
        # Ensure xGs
        print("Ensuring xG predictions...")
        self.df_season, _, _ = analyze._predict_xgs(self.df_season)
        
        # --- Fetch Schedule to get Game Dates ---
        print("Fetching schedule to resolve game dates...")
        schedule = nhl_api.get_season(team='all', season=self.season)
        
        # Create mapping: game_id -> date
        gid_to_date = {}
        for g in schedule:
            gid = g.get('id') or g.get('gamePk')
            # Date: startTimeUTC or gameDate
            date_str = g.get('gameDate') or g.get('startTimeUTC')
            if gid and date_str:
                # Parse date (YYYY-MM-DD or ISO)
                try:
                    dt = pd.to_datetime(date_str).tz_localize(None) # Remove timezone for comparison
                    gid_to_date[str(gid)] = dt
                except Exception:
                    pass
                    
        # Map to DataFrame
        # Ensure game_id in df is string for mapping
        self.df_season['game_id_str'] = self.df_season['game_id'].fillna(0).astype(int).astype(str)
        self.df_season['game_date'] = self.df_season['game_id_str'].map(gid_to_date)
        
        # Drop rows with no date
        missing_dates = self.df_season['game_date'].isna().sum()
        if missing_dates > 0:
            print(f"Warning: {missing_dates} rows have no resolved game date. Dropping them.")
            self.df_season = self.df_season.dropna(subset=['game_date'])
            
        print("Data loaded successfully.")
        
        # --- Calculate League Averages ---
        print("Calculating league averages...")
        # We need total xG and total Time for 5v5 and PP/PK across the league
        # This is expensive to do game-by-game.
        # Approximation: Use the full dataframe for xG sums, and estimate time?
        # Better: Use a sample of games or just iterate all games once (we have them loaded).
        # We can use the same logic as get_team_rates but for all teams? No, that's too slow.
        
        # Fast way:
        # Filter DF by game_state
        # 5v5 xG
        xg_5v5_total = self.df_season[self.df_season['game_state'] == '5v5']['xgs'].sum()
        
        # PP xG (5v4 and 4v5)
        # Note: 5v4 is PP for Home, PK for Away. 4v5 is PK for Home, PP for Away.
        # Total PP xG = Total xG in 5v4 + Total xG in 4v5
        xg_pp_total = self.df_season[self.df_season['game_state'].isin(['5v4', '4v5'])]['xgs'].sum()
        
        # Time?
        # We need total 5v5 time in the season.
        # We can iterate unique games and sum their 5v5 time.
        # This requires calling timing.get_game_intervals_cached for every game.
        # Might be slow if cache is cold.
        # But daily.py pre-computes this! So it should be fast if daily.py ran.
        # If not, it might take a while.
        # Let's try it.
        
        unique_game_ids = self.df_season['game_id'].unique()
        total_5v5_time = 0.0
        total_pp_time = 0.0 # Combined PP/PK time (since PP time = PK time roughly)
        
        # We only need a robust estimate.
        # Let's sample if too many games?
        # Or just do it. 1300 games * 0.01s = 13s. Acceptable.
        
        count = 0
        for gid in unique_game_ids:
            try:
                # 5v5
                intervals_5v5 = timing.get_game_intervals_cached(int(gid), self.season, {'game_state': ['5v5'], 'is_net_empty': [0]})
                total_5v5_time += sum(e-s for s,e in intervals_5v5)
                
                # PP/PK (Special Teams)
                # We want total time spent in 5v4 or 4v5.
                # Note: 5v4 and 4v5 are disjoint.
                intervals_5v4 = timing.get_game_intervals_cached(int(gid), self.season, {'game_state': ['5v4'], 'is_net_empty': [0]})
                intervals_4v5 = timing.get_game_intervals_cached(int(gid), self.season, {'game_state': ['4v5'], 'is_net_empty': [0]})
                
                total_pp_time += sum(e-s for s,e in intervals_5v4) + sum(e-s for s,e in intervals_4v5)
                
            except Exception:
                pass
            count += 1
            if count % 200 == 0:
                print(f"Processed {count} games for league averages...")
                
        # Calculate Rates (per 60 per team)
        if total_5v5_time > 0:
            self.league_xg_5v5_rate = (xg_5v5_total / (total_5v5_time / 3600)) / 2.0
        else:
            self.league_xg_5v5_rate = 2.5 # Fallback
            
        if total_pp_time > 0:
            self.league_xg_pp_rate = (xg_pp_total / (total_pp_time / 3600))
        else:
            self.league_xg_pp_rate = 6.5 # Fallback
            
        print(f"League Averages: 5v5={self.league_xg_5v5_rate:.2f}, PP={self.league_xg_pp_rate:.2f}")
        
        # --- Calculate Skill Factors ---
        self._calculate_skill_factors()

    def _calculate_skill_factors(self):
        """
        Calculate Offense and Defense Skill Factors (GAx) for all teams.
        Factors are regressed to the mean (1.0).
        """
        print("Calculating team skill factors...")
        df = self.df_season
        teams = sorted(df['home_abb'].dropna().unique())
        
        self.skill_factors = {}
        skill_weight = 0.5 # Regression weight
        
        for team in teams:
            # Get team ID
            try:
                sample = df[df['home_abb'] == team]
                if not sample.empty:
                    tid = sample.iloc[0]['home_id']
                else:
                    sample = df[df['away_abb'] == team]
                    tid = sample.iloc[0]['away_id']
            except:
                continue
                
            # Filter games involving team
            mask = (df['home_abb'] == team) | (df['away_abb'] == team)
            df_team = df[mask]
            
            # Offense
            off_mask = (df_team['team_id'] == tid)
            gf = df_team[off_mask & (df_team['event'] == 'goal')].shape[0]
            xgf = df_team[off_mask]['xgs'].sum()
            
            # Defense
            def_mask = (df_team['team_id'] != tid)
            ga = df_team[def_mask & (df_team['event'] == 'goal')].shape[0]
            xga = df_team[def_mask]['xgs'].sum()
            
            # Calculate Factors
            off_factor = 1.0
            def_factor = 1.0
            
            if xgf > 0:
                raw_off = gf / xgf
                off_factor = 1.0 + (raw_off - 1.0) * skill_weight
                
            if xga > 0:
                raw_def = ga / xga
                def_factor = 1.0 + (raw_def - 1.0) * skill_weight
                
            self.skill_factors[team] = {'off': off_factor, 'def': def_factor}
            
        print("Skill factors calculated.")

    def _get_games_for_team(self, team, date=None):
        """
        Get all games for a team up to a specific date.
        
        Args:
            team (str): Team abbreviation (e.g., 'PHI').
            date (datetime, optional): Cutoff date. If None, uses all games.
            
        Returns:
            pd.DataFrame: DataFrame of games involving the team.
        """
        # Identify team ID
        team_id = None
        # Try to find by abbreviation in home/away columns
        sample = self.df_season[self.df_season['home_abb'] == team]
        if not sample.empty:
            team_id = sample.iloc[0]['home_id']
        else:
            sample = self.df_season[self.df_season['away_abb'] == team]
            if not sample.empty:
                team_id = sample.iloc[0]['away_id']
                
        if not team_id:
            # Maybe team is already an ID?
            try:
                team_id = int(team)
            except:
                raise ValueError(f"Could not resolve team ID for {team}")
            
        # Filter games
        mask = (self.df_season['home_id'] == team_id) | (self.df_season['away_id'] == team_id)
        team_games_df = self.df_season[mask].copy()
        
        if date:
            team_games_df = team_games_df[team_games_df['game_date'] < date]
            
        return team_games_df, team_id

    def get_team_rates(self, team, date=None):
        """
        Calculate weighted xG rates (For and Against) for 5v5, PP, PK.
        
        Returns:
            dict: {
                '5v5': {'xg_for_60': float, 'xg_ag_60': float},
                '5v4': {'xg_for_60': float}, # PP Offense
                '4v5': {'xg_ag_60': float},  # PK Defense
                'games_played': int
            }
        """
        games_df, team_id = self._get_games_for_team(team, date)
        
        if games_df.empty:
            return None

        # Get unique game IDs and dates
        unique_games = games_df[['game_id', 'game_date']].drop_duplicates().sort_values('game_date')
        
        # Calculate weights based on recency
        n_games = len(unique_games)
        weights = np.exp(-self.weight_decay * np.arange(n_games)[::-1])
        
        # Helper to get TOI from cached intervals
        def get_toi(gid, season, condition):
            try:
                # timing.get_game_intervals_cached returns list of (start, end)
                intervals = timing.get_game_intervals_cached(int(gid), season, condition)
                return sum(end - start for start, end in intervals)
            except Exception as e:
                # print(f"Error getting TOI for {gid}: {e}")
                return 0.0

        # Helper to filter events
        def filter_cond(df, cond):
            mask = pd.Series(True, index=df.index)
            for k, v in cond.items():
                mask &= df[k].isin(v)
            return df[mask]

        game_stats_list = []
        
        for idx, (gid, gdate) in enumerate(zip(unique_games['game_id'], unique_games['game_date'])):
            g_df = games_df[games_df['game_id'] == gid]
            if g_df.empty: continue
            
            # Determine if team is Home or Away
            # We need to be careful with types
            row_home_id = g_df.iloc[0]['home_id']
            is_home = (str(row_home_id) == str(team_id)) or (row_home_id == team_id)
            
            # --- 5v5 ---
            cond_5v5 = {'game_state': ['5v5'], 'is_net_empty': [0]}
            t_5v5 = get_toi(gid, self.season, cond_5v5)
            
            g_5v5 = filter_cond(g_df, {'game_state': ['5v5']})
            xg_for_5v5 = g_5v5[g_5v5['team_id'] == team_id]['xgs'].sum()
            xg_ag_5v5 = g_5v5[g_5v5['team_id'] != team_id]['xgs'].sum()
            
            # Actual Goals 5v5
            gf_5v5 = g_5v5[(g_5v5['team_id'] == team_id) & (g_5v5['event'] == 'goal')].shape[0]
            ga_5v5 = g_5v5[(g_5v5['team_id'] != team_id) & (g_5v5['event'] == 'goal')].shape[0]
            
            # --- Special Teams ---
            if is_home:
                cond_pp = {'game_state': ['5v4'], 'is_net_empty': [0]}
                cond_pk = {'game_state': ['4v5'], 'is_net_empty': [0]}
            else:
                cond_pp = {'game_state': ['4v5'], 'is_net_empty': [0]}
                cond_pk = {'game_state': ['5v4'], 'is_net_empty': [0]}
                
            # PP
            t_pp = get_toi(gid, self.season, cond_pp)
            g_pp = filter_cond(g_df, {'game_state': cond_pp['game_state']})
            xg_for_pp = g_pp[g_pp['team_id'] == team_id]['xgs'].sum()
            gf_pp = g_pp[(g_pp['team_id'] == team_id) & (g_pp['event'] == 'goal')].shape[0]
            
            # PK
            t_pk = get_toi(gid, self.season, cond_pk)
            g_pk = filter_cond(g_df, {'game_state': cond_pk['game_state']})
            xg_ag_pk = g_pk[g_pk['team_id'] != team_id]['xgs'].sum()
            ga_pk = g_pk[(g_pk['team_id'] != team_id) & (g_pk['event'] == 'goal')].shape[0]
            
            game_stats_list.append({
                'weight': weights[idx],
                '5v5_xg_for': xg_for_5v5,
                '5v5_xg_ag': xg_ag_5v5,
                '5v5_gf': gf_5v5,
                '5v5_ga': ga_5v5,
                '5v5_time': t_5v5,
                'pp_xg_for': xg_for_pp,
                'pp_gf': gf_pp,
                'pp_time': t_pp,
                'pk_xg_ag': xg_ag_pk,
                'pk_ga': ga_pk,
                'pk_time': t_pk
            })
            
        df_stats = pd.DataFrame(game_stats_list)
        
        def weighted_rate(col_val, col_time):
            if df_stats.empty: return 0.0
            w_val = (df_stats[col_val] * df_stats['weight']).sum()
            w_time = (df_stats[col_time] * df_stats['weight']).sum()
            if w_time <= 0: return 0.0
            return (w_val / w_time) * 3600

        def weighted_avg_time(col_time):
            if df_stats.empty: return 0.0
            return (df_stats[col_time] * df_stats['weight']).sum() / df_stats['weight'].sum()

        results = {
            '5v5': {
                'xg_for_60': weighted_rate('5v5_xg_for', '5v5_time'),
                'xg_ag_60': weighted_rate('5v5_xg_ag', '5v5_time'),
                'gf_60': weighted_rate('5v5_gf', '5v5_time'),
                'ga_60': weighted_rate('5v5_ga', '5v5_time'),
                'avg_time': weighted_avg_time('5v5_time')
            },
            'pp': {
                'xg_for_60': weighted_rate('pp_xg_for', 'pp_time'),
                'gf_60': weighted_rate('pp_gf', 'pp_time'),
                'avg_time': weighted_avg_time('pp_time')
            },
            'pk': {
                'xg_ag_60': weighted_rate('pk_xg_ag', 'pk_time'),
                'ga_60': weighted_rate('pk_ga', 'pk_time'),
                'avg_time': weighted_avg_time('pk_time')
            },
            'games_played': n_games
        }
        
        return results

    def predict_matchup(self, home_team, away_team, date=None):
        """
        Predict the outcome of a game.
        """
        if date is None:
            date = datetime.now()
            
        print(f"Predicting {home_team} vs {away_team} (Date: {date.date()})")
        
        # Get stats
        home_stats = self.get_team_rates(home_team, date)
        away_stats = self.get_team_rates(away_team, date)
        
        if not home_stats or not away_stats:
            print("Insufficient data for prediction.")
            return None
            
        # League Averages (Approximation or calculated)
        # For Log5, we need league average xG/60.
        league_xg_5v5 = getattr(self, 'league_xg_5v5_rate', 2.5)
        league_xg_pp = getattr(self, 'league_xg_pp_rate', 6.5)
        
        # --- Skill Factors ---
        h_skill = self.skill_factors.get(home_team, {'off': 1.0, 'def': 1.0})
        a_skill = self.skill_factors.get(away_team, {'off': 1.0, 'def': 1.0})
        
        # --- 5v5 Prediction ---
        # Raw Rates
        h_5v5_rate_raw = (home_stats['5v5']['xg_for_60'] * away_stats['5v5']['xg_ag_60']) / league_xg_5v5
        a_5v5_rate_raw = (away_stats['5v5']['xg_for_60'] * home_stats['5v5']['xg_ag_60']) / league_xg_5v5
        
        # Skill Adjusted Rates
        # Home Offense * Home Off Skill
        # Away Defense * Away Def Skill
        h_5v5_rate_adj = (
            (home_stats['5v5']['xg_for_60'] * h_skill['off']) * 
            (away_stats['5v5']['xg_ag_60'] * a_skill['def'])
        ) / league_xg_5v5
        
        a_5v5_rate_adj = (
            (away_stats['5v5']['xg_for_60'] * a_skill['off']) * 
            (home_stats['5v5']['xg_ag_60'] * h_skill['def'])
        ) / league_xg_5v5
        
        # Projected 5v5 Time (Average of both teams' usual 5v5 time)
        proj_5v5_time = (home_stats['5v5']['avg_time'] + away_stats['5v5']['avg_time']) / 2.0
        
        h_5v5_xg = h_5v5_rate_adj * (proj_5v5_time / 3600)
        a_5v5_xg = a_5v5_rate_adj * (proj_5v5_time / 3600)
        
        h_5v5_xg_raw = h_5v5_rate_raw * (proj_5v5_time / 3600)
        a_5v5_xg_raw = a_5v5_rate_raw * (proj_5v5_time / 3600)
        
        # --- Special Teams Prediction ---
        # Home PP vs Away PK
        # Home PP xGF = (Home PP Off * Away PK Def) / League PP Avg
        h_pp_rate_raw = (home_stats['pp']['xg_for_60'] * away_stats['pk']['xg_ag_60']) / league_xg_pp
        
        # Apply Skill?
        # Yes. Home PP Offense Skill vs Away PK Defense Skill.
        # We use the same general "Offense" and "Defense" factors for now.
        h_pp_rate_adj = (
            (home_stats['pp']['xg_for_60'] * h_skill['off']) * 
            (away_stats['pk']['xg_ag_60'] * a_skill['def'])
        ) / league_xg_pp
        
        # Away PP vs Home PK
        a_pp_rate_raw = (away_stats['pp']['xg_for_60'] * home_stats['pk']['xg_ag_60']) / league_xg_pp
        
        a_pp_rate_adj = (
            (away_stats['pp']['xg_for_60'] * a_skill['off']) * 
            (home_stats['pk']['xg_ag_60'] * h_skill['def'])
        ) / league_xg_pp
        
        # Projected PP Times
        # Home PP Time = Avg of (Home PP Time + Away PK Time) / 2
        h_pp_time = (home_stats['pp']['avg_time'] + away_stats['pk']['avg_time']) / 2.0
        a_pp_time = (away_stats['pp']['avg_time'] + home_stats['pk']['avg_time']) / 2.0
        
        h_pp_xg = h_pp_rate_adj * (h_pp_time / 3600)
        a_pp_xg = a_pp_rate_adj * (a_pp_time / 3600)
        
        h_pp_xg_raw = h_pp_rate_raw * (h_pp_time / 3600)
        a_pp_xg_raw = a_pp_rate_raw * (a_pp_time / 3600)
        
        # --- Totals ---
        # Add Home Ice Advantage (e.g., +5% to Home xG or flat +0.1)
        # Let's use flat +0.1 xG for now
        home_ice_bonus = 0.1
        
        total_home_xg = h_5v5_xg + h_pp_xg + home_ice_bonus
        total_away_xg = a_5v5_xg + a_pp_xg
        
        total_home_xg_raw = h_5v5_xg_raw + h_pp_xg_raw + home_ice_bonus
        total_away_xg_raw = a_5v5_xg_raw + a_pp_xg_raw
        
        return PredictionResult(
            home_team, away_team,
            total_home_xg, total_away_xg,
            details={
                '5v5': (h_5v5_xg, a_5v5_xg),
                'pp': (h_pp_xg, a_pp_xg),
                'rates': (home_stats, away_stats),
                'raw_xg': (total_home_xg_raw, total_away_xg_raw),
                'skill_factors': (h_skill, a_skill),
                '5v5_raw': (h_5v5_xg_raw, a_5v5_xg_raw),
                'pp_raw': (h_pp_xg_raw, a_pp_xg_raw),
                'home_ice': home_ice_bonus
            }
        )

class PredictionResult:
    def __init__(self, home, away, home_xg, away_xg, details=None):
        self.home = home
        self.away = away
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.details = details or {}
        
    def run_simulation(self, n_sims=10000):
        """
        Run a Monte Carlo simulation of the matchup.
        """
        # Simulate scores
        h_scores = np.random.poisson(self.home_xg, n_sims)
        a_scores = np.random.poisson(self.away_xg, n_sims)
        
        # Calculate differentials (Home - Away)
        self.sim_differentials = h_scores - a_scores
        self.n_sims = n_sims
        
        # Calculate Win Probabilities from simulation
        h_wins = np.sum(self.sim_differentials > 0)
        a_wins = np.sum(self.sim_differentials < 0)
        ties = np.sum(self.sim_differentials == 0)
        
        self.sim_win_probs = {
            'home': h_wins / n_sims,
            'away': a_wins / n_sims,
            'tie': ties / n_sims
        }

    def plot(self, filename=None):
        """
        Generate a comprehensive prediction plot.
        """
        # Run simulation if not already done
        if not hasattr(self, 'sim_differentials'):
            self.run_simulation()
            
        # Use a slightly taller figure and better GridSpec spacing
        fig = plt.figure(figsize=(12, 16), facecolor='white')
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.8, 1.2, 1.5, 1.5], hspace=0.4)
        
        # Colors
        c_home = 'black'
        c_away = '#ff7f0e' # Orange
        c_tie = 'gray'
        
        # Win Probs
        h_win = self.sim_win_probs['home']
        a_win = self.sim_win_probs['away']
        h_win_total = h_win + 0.5 * self.sim_win_probs['tie']
        a_win_total = a_win + 0.5 * self.sim_win_probs['tie']
        
        # --- 1. Header / Scoreboard ---
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        
        # Title
        ax_header.text(0.5, 0.95, f"{self.home} vs {self.away}", 
                      ha='center', va='center', fontsize=24, fontweight='bold', color='black')
        
        # Scores (Adjusted)
        # Use a larger font for the main numbers
        ax_header.text(0.35, 0.7, f"{self.home}", ha='center', va='center', fontsize=18, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.7, f"{self.away}", ha='center', va='center', fontsize=18, fontweight='bold', color=c_away)
        
        ax_header.text(0.35, 0.5, f"{self.home_xg:.2f}", 
                      ha='center', va='center', fontsize=48, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.5, f"{self.away_xg:.2f}", 
                      ha='center', va='center', fontsize=48, fontweight='bold', color=c_away)
        
        ax_header.text(0.5, 0.5, "xG (Skill Adj)", ha='center', va='center', fontsize=12, color='gray', style='italic')
        
        # Win Probabilities
        ax_header.text(0.35, 0.25, f"{h_win_total*100:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.25, f"{a_win_total*100:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold', color=c_away)
        ax_header.text(0.5, 0.25, "Win Probability", ha='center', va='center', fontsize=10, color='gray')

        # --- 2. Breakdown Table ---
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')
        
        # Define column x-positions for better alignment
        col_labels = 0.15
        col_h_raw = 0.35
        col_h_adj = 0.45
        col_a_raw = 0.65
        col_a_adj = 0.75
        
        # Header Row
        y_pos = 0.9
        ax_table.text(col_labels, y_pos, "Component", ha='left', fontsize=11, fontweight='bold')
        ax_table.text(col_h_raw, y_pos, f"{self.home} Raw", ha='center', fontsize=10, fontweight='bold', color=c_home, alpha=0.6)
        ax_table.text(col_h_adj, y_pos, f"Adj", ha='center', fontsize=10, fontweight='bold', color=c_home)
        ax_table.text(col_a_raw, y_pos, f"{self.away} Raw", ha='center', fontsize=10, fontweight='bold', color=c_away, alpha=0.6)
        ax_table.text(col_a_adj, y_pos, f"Adj", ha='center', fontsize=10, fontweight='bold', color=c_away)
        
        # Draw a line under header
        ax_table.plot([0.1, 0.9], [y_pos-0.05, y_pos-0.05], color='black', linewidth=1)
        
        details = self.details
        comps = [('5v5', '5v5', '5v5_raw'), ('Power Play', 'pp', 'pp_raw')]
        skill = details.get('skill_factors', ({'off':1.0}, {'off':1.0}))
        raw_xg = details.get('raw_xg', (0.0, 0.0))
        
        y_pos -= 0.15
        for label, key_adj, key_raw in comps:
            h_adj = details[key_adj][0]
            a_adj = details[key_adj][1]
            h_raw = details[key_raw][0]
            a_raw = details[key_raw][1]
            
            ax_table.text(col_labels, y_pos, label, ha='left', fontsize=11)
            ax_table.text(col_h_raw, y_pos, f"{h_raw:.2f}", ha='center', fontsize=11, alpha=0.6)
            ax_table.text(col_h_adj, y_pos, f"{h_adj:.2f}", ha='center', fontsize=11, fontweight='bold')
            ax_table.text(col_a_raw, y_pos, f"{a_raw:.2f}", ha='center', fontsize=11, alpha=0.6)
            ax_table.text(col_a_adj, y_pos, f"{a_adj:.2f}", ha='center', fontsize=11, fontweight='bold')
            y_pos -= 0.15
            
        # Home Ice Row
        home_ice = details.get('home_ice', 0.1)
        ax_table.text(col_labels, y_pos, "Home Ice", ha='left', fontsize=11)
        ax_table.text(col_h_raw, y_pos, f"{home_ice:.2f}", ha='center', fontsize=11, alpha=0.6)
        ax_table.text(col_h_adj, y_pos, f"{home_ice:.2f}", ha='center', fontsize=11, fontweight='bold')
        ax_table.text(col_a_raw, y_pos, "0.00", ha='center', fontsize=11, alpha=0.6)
        ax_table.text(col_a_adj, y_pos, "0.00", ha='center', fontsize=11, fontweight='bold')
        y_pos -= 0.15
        
        # Skill Factor Row (Subtle)
        y_pos -= 0.05
        ax_table.text(col_labels, y_pos, "Skill Factors (Off/Def)", ha='left', fontsize=9, style='italic', color='gray')
        ax_table.text(col_h_adj, y_pos, f"{skill[0]['off']:.2f} / {skill[0]['def']:.2f}", ha='center', fontsize=9, style='italic', color='gray')
        ax_table.text(col_a_adj, y_pos, f"{skill[1]['off']:.2f} / {skill[1]['def']:.2f}", ha='center', fontsize=9, style='italic', color='gray')
        
        # Total Row
        y_pos -= 0.15
        ax_table.plot([0.1, 0.9], [y_pos+0.1, y_pos+0.1], color='black', linewidth=1)
        ax_table.text(col_labels, y_pos, "Total xG", ha='left', fontsize=12, fontweight='bold')
        ax_table.text(col_h_raw, y_pos, f"{raw_xg[0]:.2f}", ha='center', fontsize=12, fontweight='bold', alpha=0.6)
        ax_table.text(col_h_adj, y_pos, f"{self.home_xg:.2f}", ha='center', fontsize=12, fontweight='bold')
        ax_table.text(col_a_raw, y_pos, f"{raw_xg[1]:.2f}", ha='center', fontsize=12, fontweight='bold', alpha=0.6)
        ax_table.text(col_a_adj, y_pos, f"{self.away_xg:.2f}", ha='center', fontsize=12, fontweight='bold')

        # --- 3. Tale of the Tape ---
        ax_tape = fig.add_subplot(gs[2])
        
        rates = self.details.get('rates', ({}, {}))
        h_stats, a_stats = rates
        
        metrics = ['5v5 xGF/60', '5v5 xGA/60', 'PP xGF/60', 'PK xGA/60']
        
        h_vals = [h_stats['5v5']['xg_for_60'], h_stats['5v5']['xg_ag_60'], h_stats['pp']['xg_for_60'], h_stats['pk']['xg_ag_60']]
        a_vals = [a_stats['5v5']['xg_for_60'], a_stats['5v5']['xg_ag_60'], a_stats['pp']['xg_for_60'], a_stats['pk']['xg_ag_60']]
        
        h_actuals = [h_stats['5v5'].get('gf_60', 0), h_stats['5v5'].get('ga_60', 0), h_stats['pp'].get('gf_60', 0), h_stats['pk'].get('ga_60', 0)]
        a_actuals = [a_stats['5v5'].get('gf_60', 0), a_stats['5v5'].get('ga_60', 0), a_stats['pp'].get('gf_60', 0), a_stats['pk'].get('ga_60', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax_tape.bar(x - width/2, h_vals, width, label=self.home, color=c_home, alpha=0.8)
        ax_tape.bar(x + width/2, a_vals, width, label=self.away, color=c_away, alpha=0.8)
        
        for i in range(len(metrics)):
            ax_tape.plot([x[i] - width/2 - width/2.5, x[i] - width/2 + width/2.5], [h_actuals[i], h_actuals[i]], color='red', linewidth=2, zorder=10)
            ax_tape.plot([x[i] + width/2 - width/2.5, x[i] + width/2 + width/2.5], [a_actuals[i], a_actuals[i]], color='red', linewidth=2, zorder=10)
        
        ax_tape.set_ylabel('Rate per 60', fontsize=10)
        ax_tape.set_title('Tale of the Tape (Weighted Rates)\nRed Line = Actual Goals/60', fontsize=14, fontweight='bold', pad=15)
        ax_tape.set_xticks(x)
        ax_tape.set_xticklabels(metrics, fontsize=10)
        ax_tape.legend(frameon=False)
        ax_tape.grid(axis='y', alpha=0.2, linestyle='--')
        ax_tape.spines['top'].set_visible(False)
        ax_tape.spines['right'].set_visible(False)
        
        # Remove top/right spines for cleaner look
        ax_tape.spines['top'].set_visible(False)
        ax_tape.spines['right'].set_visible(False)
        
        # --- 4. Goal Differential Histogram ---
        ax_hist = fig.add_subplot(gs[3])
        
        diffs = self.sim_differentials
        unique, counts = np.unique(diffs, return_counts=True)
        freqs = counts / self.n_sims
        freq_map = dict(zip(unique, freqs))
        
        # Determine range dynamically but keep it centered and reasonable
        max_diff = max(abs(diffs.min()), abs(diffs.max()), 4)
        x_range = np.arange(-max_diff, max_diff + 1)
        y_vals = [freq_map.get(i, 0) for i in x_range]
        
        colors = []
        for i in x_range:
            if i < 0: colors.append(c_away)
            elif i > 0: colors.append(c_home)
            else: colors.append(c_tie)
            
        bars = ax_hist.bar(x_range, y_vals, color=colors, alpha=0.7)
        
        ax_hist.set_xlabel(f"Goal Differential ({self.away} ... Tie ... {self.home})", fontsize=10)
        ax_hist.set_ylabel("Probability", fontsize=10)
        ax_hist.set_title("Projected Goal Differential (Monte Carlo)", fontsize=14, fontweight='bold', pad=15)
        
        # Only label x-ticks for reasonable range
        tick_locs = [i for i in x_range if abs(i) <= 5]
        ax_hist.set_xticks(tick_locs)
        ax_hist.set_xticklabels([str(i) if i != 0 else "Tie" for i in tick_locs], fontsize=10)
        ax_hist.set_xlim(-5.5, 5.5) # Zoom in on the interesting part
        
        for bar in bars:
            height = bar.get_height()
            if height > 0.02: # Only label significant bars
                ax_hist.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height*100:.0f}%',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
        ax_hist.grid(axis='y', alpha=0.2, linestyle='--')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        
        # Final Layout Adjustment
        # plt.tight_layout() # tight_layout can sometimes mess up custom gridspec spacing
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Prediction plot saved to {filename}")
        else:
            plt.show()
        plt.close()

def calculate_home_ice_advantage(predictor):
    """
    Calculate empirical Home Ice Advantage (Mean Home xG - Mean Away xG).
    """
    print("\n--- Estimating Home Ice Advantage ---")
    df = predictor.df_season
    
    # Group by game
    games = df.groupby('game_id')
    
    diffs = []
    
    for gid, g_df in games:
        # Identify Home Team
        try:
            home_id = g_df.iloc[0]['home_id']
        except:
            continue
            
        # Calculate Total xG for Home and Away
        home_xg = g_df[g_df['team_id'] == home_id]['xgs'].sum()
        away_xg = g_df[g_df['team_id'] != home_id]['xgs'].sum()
        
        diffs.append(home_xg - away_xg)
        
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    
    print(f"Games Analyzed: {len(diffs)}")
    print(f"Mean Home Advantage (xG): {mean_diff:.4f}")
    print(f"Median Home Advantage (xG): {median_diff:.4f}")
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diff:.2f}')
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Distribution of Home Ice Advantage (Home xG - Away xG)\nMean: {mean_diff:.3f}", fontsize=14)
    plt.xlabel("xG Difference (Home - Away)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_path = 'static/home_ice_advantage.png'
    plt.savefig(out_path, dpi=100)
    print(f"Saved plot to {out_path}")
    plt.close()
    
    return mean_diff

def optimize_recency_weight(predictor):
    """
    Find optimal recency weight by minimizing Log Loss on season predictions.
    Skipping first 10 games of the season.
    """
    print("\n--- Optimizing Recency Weight ---")
    
    # Define weights to test
    weights = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    
    # Get all games sorted by date
    df = predictor.df_season
    
    # We need to ensure we have dates. The predictor adds them in load_data.
    if 'game_date' not in df.columns:
        print("Error: game_date not in DataFrame. Cannot optimize.")
        return
        
    # Group by game
    games_list = []
    for gid, g_df in df.groupby('game_id'):
        try:
            date = g_df.iloc[0]['game_date']
            home_abb = g_df.iloc[0]['home_abb']
            away_abb = g_df.iloc[0]['away_abb']
            home_id = g_df.iloc[0]['home_id']
            
            # Outcome
            # We need actual goals to determine winner.
            home_goals = g_df[(g_df['team_id'] == home_id) & (g_df['event'] == 'goal')].shape[0]
            away_goals = g_df[(g_df['team_id'] != home_id) & (g_df['event'] == 'goal')].shape[0]
            
            home_won = 1 if home_goals > away_goals else 0 # Tie/OT? Simple win/loss for now.
            
            games_list.append({
                'date': date,
                'home': home_abb,
                'away': away_abb,
                'home_won': home_won
            })
        except:
            continue
            
    # Sort by date
    games_list.sort(key=lambda x: x['date'])
    
    # Skip first 50 games of the season to let stats stabilize.
    skip_n = 50
    eval_games = games_list[skip_n:]
    
    print(f"Total Games: {len(games_list)}")
    print(f"Eval Games: {len(eval_games)} (Skipped first {skip_n})")
    
    results = {}
    
    from sklearn.metrics import log_loss
    
    for w in weights:
        print(f"Testing weight: {w}...", end='', flush=True)
        
        # Create a new predictor with this weight
        predictor.weight_decay = w
        # Clear cache
        predictor.team_stats = {} 
        
        preds = []
        actuals = []
        
        for g in eval_games:
            try:
                res = predictor.predict_matchup(g['home'], g['away'], date=g['date'])
                
                res.run_simulation(n_sims=500) # Lower sims for speed
                p_home_win = res.sim_win_probs['home'] + 0.5 * res.sim_win_probs['tie']
                
                preds.append(p_home_win)
                actuals.append(g['home_won'])
            except Exception as e:
                continue
                
        if len(preds) > 0:
            ll = log_loss(actuals, preds)
            results[w] = ll
            print(f" Log Loss: {ll:.4f}")
        else:
            print(" No predictions.")
            
    # Plot
    if results:
        ws = list(results.keys())
        losses = list(results.values())
        
        best_w = ws[np.argmin(losses)]
        print(f"Optimal Weight: {best_w}")
        
        plt.figure(figsize=(8, 5))
        plt.plot(ws, losses, marker='o', linestyle='-', color='purple')
        plt.axvline(best_w, color='green', linestyle='--', label=f'Optimal: {best_w}')
        plt.title("Recency Weight Optimization (Log Loss)", fontsize=14)
        plt.xlabel("Weight Decay Factor")
        plt.ylabel("Log Loss (Lower is Better)")
        plt.legend()
        plt.grid(alpha=0.3)
        
        out_path = 'static/recency_optimization.png'
        plt.savefig(out_path, dpi=100)
        print(f"Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    # Initialize Predictor
    pred = GamePredictor(season='20252026')
    
    # 1. Estimate Home Ice Advantage
    #calculate_home_ice_advantage(pred)
    
    # 2. Optimize Recency Weight
    #optimize_recency_weight(pred)
    
    # 3. Run Test Prediction (using default or optimal?)
    # Let's just run the standard test for now
    print("\n--- Running Test Prediction (Default Weight) ---")
    pred.weight_decay = 0.02 # Reset to default
    res = pred.predict_matchup('PHI', 'COL', date=pd.to_datetime('2025-12-04'))
    print(f"Prediction: {res.home} {res.home_xg:.2f} - {res.away} {res.away_xg:.2f}")
    res.plot('test_prediction.png')
