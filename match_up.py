import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import timing
import analyze
from scipy.stats import poisson
from datetime import datetime
import nhl_api

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
        # Wait. League Average xG/60.
        # Is this "per game" or "per 60 minutes of play"?
        # Usually "League Average 5v5 xG/60" means:
        # (Total League 5v5 xG) / (Total League 5v5 Time / 60)
        # But wait, in a game there are 2 teams.
        # Does 2.5 xG/60 mean "Both teams combined generate 2.5"?
        # Or "One team generates 2.5"?
        # Usually it's per team.
        # Total xG is for BOTH teams. Total Time is shared.
        # So (Total xG) / (Total Time / 60) = Combined Rate.
        # Per Team Rate = Combined Rate / 2.
        
        if total_5v5_time > 0:
            self.league_xg_5v5_rate = (xg_5v5_total / (total_5v5_time / 3600)) / 2.0
        else:
            self.league_xg_5v5_rate = 2.5 # Fallback
            
        if total_pp_time > 0:
            # PP Rate.
            # Total PP xG / Total PP Time * 60.
            # This is the rate of xG generated during PP time.
            # Since in PP time, only one team is on PP (usually), the "Combined Rate" is basically the PP Team's rate + PK Team's rate (which is low).
            # But we want "League Average PP xGF/60".
            # Total PP xG includes SHG? Yes.
            # Usually we want xGF/60 for the PP team.
            # Let's assume Total PP xG is mostly PP xGF.
            # And Total PP Time is the duration of uneven strength.
            # So (Total xG) / (Total Time / 60) is roughly the PP xGF/60 (plus a bit of SH xG).
            # We don't divide by 2 because only one team is "on offense" (PP).
            self.league_xg_pp_rate = (xg_pp_total / (total_pp_time / 3600))
        else:
            self.league_xg_pp_rate = 6.5 # Fallback
            
        print(f"League Averages: 5v5={self.league_xg_5v5_rate:.2f}, PP={self.league_xg_pp_rate:.2f}")

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
            
            # --- Special Teams ---
            # If Team is Home:
            #   PP = 5v4 (Home 5, Away 4)
            #   PK = 4v5 (Home 4, Away 5)
            # If Team is Away:
            #   PP = 4v5 (Home 4, Away 5) -> Wait. '4v5' usually means Home 4, Away 5.
            #   So if Away team is on PP, they have 5 skaters, Home has 4. That is '4v5'.
            #   PK = 5v4 (Home 5, Away 4). Away team is PK (4 skaters).
            
            if is_home:
                cond_pp = {'game_state': ['5v4'], 'is_net_empty': [0]}
                cond_pk = {'game_state': ['4v5'], 'is_net_empty': [0]}
            else:
                cond_pp = {'game_state': ['4v5'], 'is_net_empty': [0]}
                cond_pk = {'game_state': ['5v4'], 'is_net_empty': [0]}
                
            # PP
            t_pp = get_toi(gid, self.season, cond_pp)
            # Filter events for PP xG For
            g_pp = filter_cond(g_df, {'game_state': cond_pp['game_state']})
            xg_for_pp = g_pp[g_pp['team_id'] == team_id]['xgs'].sum()
            
            # PK
            t_pk = get_toi(gid, self.season, cond_pk)
            # Filter events for PK xG Against
            g_pk = filter_cond(g_df, {'game_state': cond_pk['game_state']})
            xg_ag_pk = g_pk[g_pk['team_id'] != team_id]['xgs'].sum()
            
            game_stats_list.append({
                'weight': weights[idx],
                '5v5_xg_for': xg_for_5v5,
                '5v5_xg_ag': xg_ag_5v5,
                '5v5_time': t_5v5,
                'pp_xg_for': xg_for_pp,
                'pp_time': t_pp,
                'pk_xg_ag': xg_ag_pk,
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
                'avg_time': weighted_avg_time('5v5_time')
            },
            'pp': {
                'xg_for_60': weighted_rate('pp_xg_for', 'pp_time'),
                'avg_time': weighted_avg_time('pp_time')
            },
            'pk': {
                'xg_ag_60': weighted_rate('pk_xg_ag', 'pk_time'),
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
        
        # --- 5v5 Prediction ---
        # Home xGF = (Home Off * Away Def) / League Avg
        h_5v5_rate = (home_stats['5v5']['xg_for_60'] * away_stats['5v5']['xg_ag_60']) / league_xg_5v5
        a_5v5_rate = (away_stats['5v5']['xg_for_60'] * home_stats['5v5']['xg_ag_60']) / league_xg_5v5
        
        # Projected 5v5 Time (Average of both teams' usual 5v5 time)
        proj_5v5_time = (home_stats['5v5']['avg_time'] + away_stats['5v5']['avg_time']) / 2.0
        
        h_5v5_xg = h_5v5_rate * (proj_5v5_time / 3600)
        a_5v5_xg = a_5v5_rate * (proj_5v5_time / 3600)
        
        # --- Special Teams Prediction ---
        # Home PP vs Away PK
        # Home PP xGF = (Home PP Off * Away PK Def) / League PP Avg
        h_pp_rate = (home_stats['pp']['xg_for_60'] * away_stats['pk']['xg_ag_60']) / league_xg_pp
        
        # Away PP vs Home PK
        a_pp_rate = (away_stats['pp']['xg_for_60'] * home_stats['pk']['xg_ag_60']) / league_xg_pp
        
        # Projected PP Times
        # Home PP Time = Avg of (Home PP Time + Away PK Time) / 2
        h_pp_time = (home_stats['pp']['avg_time'] + away_stats['pk']['avg_time']) / 2.0
        a_pp_time = (away_stats['pp']['avg_time'] + home_stats['pk']['avg_time']) / 2.0
        
        h_pp_xg = h_pp_rate * (h_pp_time / 3600)
        a_pp_xg = a_pp_rate * (a_pp_time / 3600)
        
        # --- Totals ---
        # Add Home Ice Advantage (e.g., +5% to Home xG or flat +0.1)
        # Let's use flat +0.1 xG for now
        home_ice_bonus = 0.1
        
        total_home_xg = h_5v5_xg + h_pp_xg + home_ice_bonus
        total_away_xg = a_5v5_xg + a_pp_xg
        
        return PredictionResult(
            home_team, away_team,
            total_home_xg, total_away_xg,
            details={
                '5v5': (h_5v5_xg, a_5v5_xg),
                'pp': (h_pp_xg, a_pp_xg),
                'rates': (home_stats, away_stats)
            }
        )

class PredictionResult:
    def __init__(self, home, away, home_xg, away_xg, details=None):
        self.home = home
        self.away = away
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.details = details or {}
        
    def calculate_probabilities(self, max_goals=10):
        """
        Calculate win probabilities using Poisson distribution.
        """
        # Probability matrices
        h_probs = [poisson.pmf(i, self.home_xg) for i in range(max_goals+1)]
        a_probs = [poisson.pmf(i, self.away_xg) for i in range(max_goals+1)]
        
        home_win = 0.0
        away_win = 0.0
        tie = 0.0
        
        for h in range(max_goals+1):
            for a in range(max_goals+1):
                prob = h_probs[h] * a_probs[a]
                if h > a:
                    home_win += prob
                elif a > h:
                    away_win += prob
                else:
                    tie += prob
                    
        # Distribute tie probability (OT/SO)
        # Simple assumption: 50/50 split for OT
        home_win_final = home_win + (tie * 0.5)
        away_win_final = away_win + (tie * 0.5)
        
        return home_win_final, away_win_final, tie

    def plot(self, out_path='prediction.png'):
        """
        Generate a visual summary of the prediction.
        """
        h_win, a_win, tie = self.calculate_probabilities()
        
        # Colors matching plot.py (Home=Black, Away=Orange)
        c_home = 'black'
        c_away = 'orange'
        
        fig = plt.figure(figsize=(10, 12)) # Taller for better spacing
        gs = fig.add_gridspec(4, 1, height_ratios=[0.8, 0.2, 1, 1])
        
        # 1. Header / Scoreboard
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        
        # Title
        ax_header.text(0.5, 0.9, f"{self.home} vs {self.away}", 
                      ha='center', va='center', fontsize=20, fontweight='bold', color='black')
        
        # Scores
        ax_header.text(0.35, 0.6, f"{self.home}", ha='center', va='center', fontsize=16, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.6, f"{self.away}", ha='center', va='center', fontsize=16, fontweight='bold', color=c_away)
        
        ax_header.text(0.35, 0.4, f"{self.home_xg:.2f}", 
                      ha='center', va='center', fontsize=36, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.4, f"{self.away_xg:.2f}", 
                      ha='center', va='center', fontsize=36, fontweight='bold', color=c_away)
        
        ax_header.text(0.5, 0.4, "xG", ha='center', va='center', fontsize=12, color='gray')
        
        # Win Probabilities
        ax_header.text(0.35, 0.2, f"{h_win*100:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold', color=c_home)
        ax_header.text(0.65, 0.2, f"{a_win*100:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold', color=c_away)
        ax_header.text(0.5, 0.2, "Win %", ha='center', va='center', fontsize=10, color='gray')

        # 2. Breakdown Table (Text)
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')
        
        # Simple text table for details
        # Columns: Component | Home xG | Away xG
        y_pos = 0.8
        ax_table.text(0.2, y_pos, "Component", ha='left', fontsize=10, fontweight='bold')
        ax_table.text(0.5, y_pos, f"{self.home}", ha='center', fontsize=10, fontweight='bold', color=c_home)
        ax_table.text(0.8, y_pos, f"{self.away}", ha='center', fontsize=10, fontweight='bold', color=c_away)
        
        details = self.details
        comps = [('5v5', '5v5'), ('Power Play', 'pp')]
        
        y_pos -= 0.25
        for label, key in comps:
            h_val = details[key][0]
            a_val = details[key][1]
            ax_table.text(0.2, y_pos, label, ha='left', fontsize=10)
            ax_table.text(0.5, y_pos, f"{h_val:.2f}", ha='center', fontsize=10)
            ax_table.text(0.8, y_pos, f"{a_val:.2f}", ha='center', fontsize=10)
            y_pos -= 0.25

        # 3. Tale of the Tape (Bar Chart)
        ax_tape = fig.add_subplot(gs[2])
        
        rates = self.details.get('rates', ({}, {}))
        h_stats, a_stats = rates
        
        # Metrics to compare
        metrics = ['5v5 xGF/60', '5v5 xGA/60', 'PP xGF/60', 'PK xGA/60']
        h_vals = [
            h_stats['5v5']['xg_for_60'],
            h_stats['5v5']['xg_ag_60'],
            h_stats['pp']['xg_for_60'],
            h_stats['pk']['xg_ag_60']
        ]
        a_vals = [
            a_stats['5v5']['xg_for_60'],
            a_stats['5v5']['xg_ag_60'],
            a_stats['pp']['xg_for_60'],
            a_stats['pk']['xg_ag_60']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax_tape.bar(x - width/2, h_vals, width, label=self.home, color=c_home, alpha=0.8)
        ax_tape.bar(x + width/2, a_vals, width, label=self.away, color=c_away, alpha=0.8)
        
        ax_tape.set_xticks(x)
        ax_tape.set_xticklabels(metrics)
        ax_tape.set_title("Tale of the Tape (Weighted Rates)", fontsize=12, fontweight='bold')
        ax_tape.legend()
        ax_tape.grid(axis='y', alpha=0.3)
        
        # Remove top/right spines for cleaner look
        ax_tape.spines['top'].set_visible(False)
        ax_tape.spines['right'].set_visible(False)
        
        # 4. Uncertainty / Score Distribution
        ax_dist = fig.add_subplot(gs[3])
        
        max_g = 8
        goals = np.arange(max_g + 1)
        h_pmf = poisson.pmf(goals, self.home_xg)
        a_pmf = poisson.pmf(goals, self.away_xg)
        
        ax_dist.plot(goals, h_pmf, 'o-', color=c_home, label=f"{self.home} Dist")
        ax_dist.plot(goals, a_pmf, 'o-', color=c_away, label=f"{self.away} Dist")
        
        ax_dist.fill_between(goals, h_pmf, alpha=0.1, color=c_home)
        ax_dist.fill_between(goals, a_pmf, alpha=0.1, color=c_away)
        
        ax_dist.set_title("Score Probability Distribution", fontsize=12, fontweight='bold')
        ax_dist.set_xlabel("Goals Scored")
        ax_dist.set_ylabel("Probability")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Remove top/right spines
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Prediction plot saved to {out_path}")

if __name__ == "__main__":
    # Test Routine
    print("Running Test Prediction...")
    predictor = GamePredictor(weight_decay=0.05)
    
    # Predict a hypothetical game
    res = predictor.predict_matchup('PHI', 'PIT')
    
    if res:
        print(f"Prediction: {res.home} {res.home_xg:.2f} - {res.away} {res.away_xg:.2f}")
        res.plot('test_prediction.png')
