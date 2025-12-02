
import pandas as pd
import numpy as np
import timing
import analyze
import nhl_api
import parse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os

class QCSuite:
    def __init__(self, season: str = '20252026', out_dir: str = 'qc_report'):
        self.season = season
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        print(f"Loading season data for {season}...")
        self.df = timing.load_season_df(season)
        if self.df.empty:
            print("WARNING: No season data found.")
        
        self.report_lines = []
        self.report_lines.append(f"# Data Quality Report: Season {season}")
        self.report_lines.append("")

    def run_all(self):
        print("\n=== Starting QC Suite ===")
        self.check_special_teams_outliers()
        self.check_long_shifts()
        self.check_long_game_state_segments()
        self.check_missing_shift_data()
        
        self.save_report()
        print("\n=== QC Suite Complete ===")

    def save_report(self):
        report_path = os.path.join(self.out_dir, 'DATA_QUALITY_REPORT.md')
        with open(report_path, 'w') as f:
            f.write("\n".join(self.report_lines))
        print(f"Report saved to {report_path}")

    def add_figure(self, fig, filename, caption):
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        self.report_lines.append(f"![{caption}]({filename})")
        self.report_lines.append(f"*{caption}*")
        self.report_lines.append("")

    def check_special_teams_outliers(self, threshold_minutes: float = 20.0):
        """
        Check for games where a team has excessive time in 5v4 or 4v5.
        """
        print(f"\n[Check] Special Teams Outliers (Threshold: >{threshold_minutes} mins)")
        self.report_lines.append("## Special Teams Outliers")
        self.report_lines.append(f"Checking for games with >{threshold_minutes} minutes of 5v4 time.")
        
        teams = sorted(pd.concat([self.df['home_abb'], self.df['away_abb']]).unique())
        # Limit to 5 teams for demo speed, or all for full run. 
        # Let's do all but optimize by grouping? No, compute_game_timing needs per-game.
        # Let's sample 5 random teams to keep it fast for now.
        sample_teams = np.random.choice(teams, size=min(len(teams), 5), replace=False)
        print(f"Checking sample of {len(sample_teams)} teams: {sample_teams}")
        
        all_times = []
        outliers = []
        
        for team in sample_teams:
            cond = {'team': team, 'game_state': ['5v4'], 'is_net_empty': [0]}
            team_df = self.df[(self.df['home_abb'] == team) | (self.df['away_abb'] == team)]
            
            res = timing.compute_game_timing(team_df, cond)
            per_game = res.get('per_game', {})
            
            for gid, g_res in per_game.items():
                sec = g_res.get('intersection_seconds', 0.0)
                mins = sec / 60.0
                all_times.append(mins)
                if mins > threshold_minutes:
                    outliers.append({
                        'team': team,
                        'game_id': gid,
                        'condition': '5v4',
                        'minutes': mins
                    })
        
        # Plot Histogram
        if all_times:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_times, bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of 5v4 Time per Game (Sampled Teams)')
            ax.set_xlabel('Minutes')
            ax.set_ylabel('Frequency')
            self.add_figure(fig, 'special_teams_hist.png', 'Distribution of 5v4 Time')
        
        if outliers:
            self.report_lines.append(f"Found {len(outliers)} outliers:")
            for o in outliers:
                self.report_lines.append(f"- **{o['team']}** (Game {o['game_id']}): {o['minutes']:.2f} mins")
        else:
            self.report_lines.append("No outliers found in the sampled teams.")

    def check_long_shifts(self, threshold_seconds: float = 300.0):
        """
        Check for individual shifts longer than threshold (default 5 mins).
        Excludes goalies.
        """
        print(f"\n[Check] Long Shifts (Threshold: >{threshold_seconds}s)")
        self.report_lines.append("## Long Shifts")
        self.report_lines.append(f"Checking for skater shifts >{threshold_seconds} seconds.")
        
        gids = self.df['game_id'].unique()
        sample_gids = np.random.choice(gids, size=min(len(gids), 20), replace=False)
        
        long_shifts = []
        all_durations = []
        
        for gid in sample_gids:
            try:
                shifts = timing._get_shifts_df(int(gid))
                if shifts.empty:
                    continue
                
                roles = timing._classify_player_roles(shifts).get('roles', {})
                
                for _, row in shifts.iterrows():
                    dur = row.get('end_total_seconds', 0) - row.get('start_total_seconds', 0)
                    pid = str(row.get('player_id'))
                    if roles.get(pid) != 'G':
                        all_durations.append(dur)
                        if dur > threshold_seconds:
                            long_shifts.append({
                                'game_id': gid,
                                'player_id': pid,
                                'duration': dur
                            })
            except Exception:
                pass
                
        # Plot Histogram
        if all_durations:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_durations, bins=50, range=(0, 600), log=True, color='salmon', edgecolor='black')
            ax.set_title('Distribution of Skater Shift Durations (Log Scale)')
            ax.set_xlabel('Seconds')
            ax.set_ylabel('Frequency (Log)')
            self.add_figure(fig, 'shift_duration_hist.png', 'Shift Duration Distribution')

        if long_shifts:
            self.report_lines.append(f"Found {len(long_shifts)} long shifts in sample:")
            for s in long_shifts[:10]:
                self.report_lines.append(f"- Game {s['game_id']}, Player {s['player_id']}: {s['duration']:.1f}s")
        else:
            self.report_lines.append("No suspicious long shifts found in sample.")

    def check_long_game_state_segments(self, threshold_seconds: float = 600.0):
        """
        Check for continuous game state segments (e.g. 5v4) lasting longer than threshold.
        """
        print(f"\n[Check] Long Game State Segments (Threshold: >{threshold_seconds}s)")
        self.report_lines.append("## Long Game State Segments")
        self.report_lines.append(f"Checking for continuous 5v4/4v5 segments >{threshold_seconds} seconds.")
        
        gids = self.df['game_id'].unique()
        sample_gids = np.random.choice(gids, size=min(len(gids), 20), replace=False)
        
        suspicious_segments = []
        
        for gid in sample_gids:
            for state in ['5v4', '4v5']:
                cond = {'game_state': [state]}
                res = timing.compute_intervals_for_game(int(gid), cond)
                intervals = res.get('intervals_per_condition', {}).get('game_state', [])
                
                for start, end in intervals:
                    dur = end - start
                    if dur > threshold_seconds:
                        suspicious_segments.append({
                            'game_id': gid,
                            'state': state,
                            'duration': dur
                        })
                        
        if suspicious_segments:
            self.report_lines.append(f"Found {len(suspicious_segments)} suspicious segments:")
            for s in suspicious_segments:
                self.report_lines.append(f"- Game {s['game_id']} ({s['state']}): {s['duration']:.1f}s")
        else:
            self.report_lines.append("No suspicious segments found in sample.")

    def check_missing_shift_data(self):
        """
        Identify games with empty or minimal shift data.
        """
        print(f"\n[Check] Missing Shift Data")
        self.report_lines.append("## Missing Shift Data")
        
        gids = self.df['game_id'].unique()
        sample_gids = np.random.choice(gids, size=min(len(gids), 50), replace=False)
        
        missing = []
        for gid in sample_gids:
            shifts = timing._get_shifts_df(int(gid))
            if shifts.empty or len(shifts) < 10:
                missing.append(gid)
                
        if missing:
            self.report_lines.append(f"Found {len(missing)} games with missing/minimal shift data:")
            for m in missing:
                self.report_lines.append(f"- {m}")
        else:
            self.report_lines.append("No missing shift data found in sample.")

if __name__ == "__main__":
    qc = QCSuite()
    qc.run_all()
