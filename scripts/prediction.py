"""scripts/prediction.py

Predicts Team Goals For (GF) and Goals Against (GA) per 60 minutes.
Combines:
1. Base xG Rates (per game state)
2. Team Skill Proxies (Offense/Defense Multipliers per state)
3. TOI Distributions (Adjustable)

Outputs:
- Visualization: analysis/prediction_scatter.png (League-wide)
- Visualization: analysis/prediction_breakdown_{TEAM}.png (Per-team breakdown)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, analyze

class TeamPredictor:
    def __init__(self, season="20252026"):
        self.season = season
        self.skill_proxies = None
        self.base_stats = None
        self.teams_list = []
        
        self._load_data()
        
    def _load_data(self):
        """Loads skill proxies and calculates base xG rates from season data."""
        # 1. Load Skill Proxies
        proxy_path = Path('analysis/team_skill_proxies.csv')
        if not proxy_path.exists():
            raise FileNotFoundError(f"Skill proxies not found at {proxy_path}. Run quantify_team_skill.py first.")
        
        self.skill_proxies = pd.read_csv(proxy_path)
        # Ensure TeamID is string
        self.skill_proxies['TeamID'] = self.skill_proxies['TeamID'].astype(str).str.replace(r'\.0$', '', regex=True)
        self.teams_list = self.skill_proxies['TeamAbbrev'].unique().tolist()
        
        # 2. Load Season Data for Base Rates
        data_path = Path(f"data/{self.season}/{self.season}_df.csv")
        if not data_path.exists():
            data_path = Path(f"data/{self.season}.csv")
            
        logger.info(f"Loading season data from {data_path}...")
        df = fit_xgs.load_data(str(data_path))
        
        # Ensure predictions exist
        model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
        if not model_path.exists():
            model_path = Path('analysis/xgs/xg_model_nested.joblib')
            
        if 'xgs' not in df.columns:
            logger.info("Generating xG predictions...")
            df, _, _ = analyze._predict_xgs(df, model_path=str(model_path))
            
        if 'home_skaters' not in df.columns or 'away_skaters' not in df.columns:
            df['home_skaters'] = 5
            df['away_skaters'] = 5
            
        # Ensure is_goal
        if 'is_goal' not in df.columns:
            df['is_goal'] = (df['event'] == 'goal').astype(int)
            
        # Calculate Base Stats per Team per State
        logger.info("Calculating base xG rates and TOI defaults...")
        self.base_stats = {}
        
        # Map state to label
        def get_state_label(h_n, a_n, is_home):
            # Return 5v5, 5v4, 4v5 based on team perspective
            my_n = h_n if is_home else a_n
            opp_n = a_n if is_home else h_n
            
            if my_n == 5 and opp_n == 5: return '5v5'
            if my_n == 5 and opp_n == 4: return '5v4' # PP
            if my_n == 4 and opp_n == 5: return '4v5' # SH
            return 'Other'

        # We need to iterate teams to get their specific stats
        # Doing this efficiently: group by home/away team + state
        
        # Assign State Labels relative to Home
        # 5v5, Home_5v4 (Home PP), Home_4v5 (Home SH)
        df['Home_State'] = df.apply(lambda r: get_state_label(r['home_skaters'], r['away_skaters'], True), axis=1)
        df['Away_State'] = df.apply(lambda r: get_state_label(r['home_skaters'], r['away_skaters'], False), axis=1)
        
        # Filter to relevant events for time (period_end, etc? No, use total time estimates or just sum events?)
        # Better: Use `analysis.timing`? Or just aggregate xG and approximate TOI from events?
        # Accurate TOI is hard from just events. 
        # For xGF_Rate = xG / Time, we need Time.
        # Let's approximate Time by: Events * (Avg Seconds/Event). Or load league stats w/ TOI?
        # Let's use `run_league_stats` output? That has TOI per state per team.
        # That's cleaner.
        
        # ACTUALLY: Let's calculate xG Sums from this DF, and load TOI from `analysis/league/{season}/{state}/season_team_summary.json`?
        # Or simpler: Just calculate xG Sums here, and assume a standard Time per Event? NO, bad.
        
        # Let's stick to using the `prediction.py` to aggregate what we have loaded.
        # We'll calculate xG Sums. For TOI, we can pull from the `game_state` frequency.
        # Total Game Time ~ 3600s.
        # Fraction of events in state ~ Fraction of Time? Roughly.
        
        for tid in self.skill_proxies['TeamID'].astype(str):
            team_stats = {'xGF': {}, 'xGA': {}, 'Count': {}, 'TotalEvents': 0}
            
            # Filter rows involved
            # This is slow if we loop 32 times on full DF.
            # Faster: Groupby TeamID after melting?
            pass
        
        # Fast Path: Melt to (TeamID, State, Type, xG)
        # Type: For/Against
        home_df = df[['home_id', 'Home_State', 'xgs', 'is_goal']].rename(
            columns={'home_id': 'TeamID', 'Home_State': 'State'}
        )
        home_df['Type'] = 'Home'
        
        away_df = df[['away_id', 'Away_State', 'xgs', 'is_goal']].rename(
            columns={'away_id': 'TeamID', 'Away_State': 'State'}
        )
        away_df['Type'] = 'Away'
        
        # Concat
        melted = pd.concat([home_df, away_df])
        melted['TeamID'] = melted['TeamID'].fillna(-1).astype(int).astype(str)
        
        # Groupby
        grouped = melted.groupby(['TeamID', 'State', 'Type']).agg(
            xG_Sum=('xgs', 'sum'),
            Events=('xgs', 'count')
        ).reset_index()
        
        # Process into self.base_stats
        # Need Total Events per Team to est. TOI distribution
        totals = grouped.groupby('TeamID')['Events'].sum().to_dict()
        
        for tid, group in grouped.groupby('TeamID'):
            if tid == '-1': continue
            
            self.base_stats[tid] = {
                'rates': {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0}, # xG per "Event" or per 60?
                'def_rates': {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0},
                'toi': {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0}
            }
            
            total_evs = totals.get(tid, 1)
            
            # Events -> TOI approximation
            # Avg NHL game: ~60 mins. ~100-120 Corsi events? ~200 total events?
            # Let's just use Event Share as TOI Share for now.
            # And calculate xG per Event.
            
            for _, row in group.iterrows():
                st = row['State']
                if st not in ['5v5', '5v4', '4v5']: continue
                
                # TOI Share
                # We only sum 5v5, 5v4, 4v5 events for the "Known Universe" share?
                # Or relative to Total?
                # Let's normalize later.
                
                # xG per Event (Base Rate)
                xg_per_event = row['xG_Sum'] / row['Events'] if row['Events'] > 0 else 0
                
                if row['Type'] == 'Home': 
                    # Home=For? Wait. 'Type' came from source DF.
                    # home_df -> Type='Home'. 
                    # If I am Home Team, xG is For.
                    self.base_stats[tid]['rates'][st] = xg_per_event
                    
                    # Store event count for TOI calc
                    # But need to distinguish For/Against events?
                    # Events happen TO/AGAINST a team. They share the same clock.
                    # Total Events in State = (Home_Events_In_State + Away_Events_In_State)? 
                    # No, melting duplicated the events (once for home, once for away).
                    # So 'Events' for 'TeamID' in 'State' is all events that occurred while they were in that state.
                    self.base_stats[tid]['toi'][st] += row['Events']
                    
                else: 
                    # Type='Away'. If I am Away Team, xG is For?
                    # Yes, melted structure:
                    # home_df: TeamID=HomeID, xG=xG. xG is generated BY that team?
                    # Wait, in the DF, 'xgs' is for the EVENT.
                    # We need to correctly attribute xGF and xGA.
                    pass
    
    def _load_data_refined(self):
        """Correct Logic for Base Rates and TOI."""
        # Reload proxies
        proxy_path = Path('analysis/team_skill_proxies.csv')
        self.skill_proxies = pd.read_csv(proxy_path)
        self.skill_proxies['TeamID'] = self.skill_proxies['TeamID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        # Load Data
        data_path = Path(f"data/{self.season}/{self.season}_df.csv")
        if not data_path.exists():
            data_path = Path(f"data/{self.season}.csv")
            
        df = fit_xgs.load_data(str(data_path))
        
        # Ensure preds
        model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
        if not model_path.exists(): model_path = Path('analysis/xgs/xg_model_nested.joblib')
        if 'xgs' not in df.columns:
            logger.info("Generating predictions...")
            df, _, _ = analyze._predict_xgs(df, model_path=str(model_path))
            
        # Parse States
        if 'game_state' in df.columns:
             def parse(s):
                 if not isinstance(s, str) or 'v' not in s: return 5, 5
                 try: return int(s.split('v')[0]), int(s.split('v')[1])
                 except: return 5, 5
             parsed = df['game_state'].apply(parse)
             df['h_sk'], df['a_sk'] = parsed.str[0], parsed.str[1]
        else:
             df['h_sk'], df['a_sk'] = 5, 5
             
        # Attribution
        # Shot Events only for xG
        shots = df[df['event'].isin(['shot-on-goal', 'blocked-shot', 'missed-shot', 'goal'])].copy()
        
        # Determine Shooting/Defending Team
        # (Simplified logic)
        shots['shooting_team'] = shots['team_id'] # blocked shots handled below?
        shots['defending_team'] = np.where(shots['team_id'] == shots['home_id'], shots['away_id'], shots['home_id'])
        
        # Fix Blocked: Event Team is DEFENDER
        mask_blk = shots['event'] == 'blocked-shot'
        shots.loc[mask_blk, 'defending_team'] = shots.loc[mask_blk, 'team_id']
        shots.loc[mask_blk, 'shooting_team'] = np.where(shots.loc[mask_blk, 'team_id'] == shots.loc[mask_blk, 'home_id'],
                                                       shots.loc[mask_blk, 'away_id'], shots.loc[mask_blk, 'home_id'])
                                                       
        # Assign State relative to Shooter
        # If Shooter is Home: State is H_Sk v A_Sk
        # If Shooter is Away: State is A_Sk v H_Sk
        mask_shoot_home = shots['shooting_team'] == shots['home_id']
        shots['s_sk'] = np.where(mask_shoot_home, shots['h_sk'], shots['a_sk'])
        shots['d_sk'] = np.where(mask_shoot_home, shots['a_sk'], shots['h_sk'])
        
        def get_st(s, d):
            if s==5 and d==5: return '5v5'
            if s==5 and d==4: return '5v4'
            if s==4 and d==5: return '4v5'
            return 'Other'
            
        shots['State'] = shots.apply(lambda r: get_st(r['s_sk'], r['d_sk']), axis=1)
        
        # Aggregate xGF per Team per State
        # xGA is just the reverse
        
        # TOI Estimate:
        # We need total time spent in each state per team.
        # We can sum `time_elapsed`? No, that's cumulative. 
        # `period_time` differences? Hard.
        # APPROXIMATION: 
        # Assume 60 mins total.
        # Distribution = (Count of ALL events in state) / (Total Events).
        # We use ALL events df for TOI, not just shots.
        
        # Events DF for TOI
        df['State_Home'] = df.apply(lambda r: get_st(r['h_sk'], r['a_sk']), axis=1)
        df['State_Away'] = df.apply(lambda r: get_st(r['a_sk'], r['h_sk']), axis=1)
        
        # Count events per team per state
        # Home Team gets +1 event for State_Home
        # Away Team gets +1 event for State_Away
        
        toi_counts = {}
        for tid in self.skill_proxies['TeamID'].astype(str):
            toi_counts[tid] = {'5v5': 0, '5v4': 0, '4v5': 0, 'Other': 0}
            
        # Optimization: Value Counts on Home/Away subsets
        h_counts = df.groupby(['home_id', 'State_Home']).size().reset_index(name='count')
        for _, r in h_counts.iterrows():
            tid = str(int(r['home_id']))
            st = r['State_Home']
            if tid in toi_counts and st in toi_counts[tid]:
                toi_counts[tid][st] += r['count']
                
        a_counts = df.groupby(['away_id', 'State_Away']).size().reset_index(name='count')
        for _, r in a_counts.iterrows():
            tid = str(int(r['away_id']))
            st = r['State_Away']
            if tid in toi_counts and st in toi_counts[tid]:
                toi_counts[tid][st] += r['count']
                
        # Calculate Base xG Rates (xG per Event in State)
        # Using `shots` df
        xg_sums = shots.groupby(['shooting_team', 'State'])['xgs'].sum().to_dict()
        xga_sums = shots.groupby(['defending_team', 'State'])['xgs'].sum().to_dict()
        
        self.base_stats = {}
        target_states = ['5v5', '5v4', '4v5']
        
        for tid in self.skill_proxies['TeamID'].astype(str):
            tid_float = float(tid) # Groupby uses floats/ints often
            
            # Normalize TOI to fraction
            total_ev = sum(toi_counts[tid].values())
            if total_ev == 0: total_ev = 1
            
            dist = {k: v/total_ev for k,v in toi_counts[tid].items()}
            
            # Normalize to exclude 'Other' if we only predict main states?
            # Or keep 'Other' component?
            # Let's normalize the 3 main states to sum to (1 - Other_Fraction)?
            # Or just normalize these 3 to 1.0 for the prediction mix if user provides?
            # Standard prediction: xG_Rate * 60.
            # xG_Rate = xG / Time.
            # Time_State = Total_Time * Dist_State.
            # So Base_xG_60_State = (Sum_xG_State) / (Total_Time * Dist_State) * 60.
            # This cancels out Total_Time? 
            # No. Base xG/60 is intrinsic.
            # Let's calculate xG / Event * Events/60.
            # Avg Events/60 ~ 120 approx? 
            # Let's purely predict relative to LEAGUE AVERAGE xG/60 for that state, adjusted by team?
            # No, we want Team's specific xG generation.
            
            # Simpler:
            # 1. Calculate Team's xG per Event in State.
            # 2. Assume standard "Pace" (Events per 60 mins). Say 100 events/60.
            # 3. Predict = (xG/Event) * Skill * (Events/60 * Dist_State).
            
            # Let's use the actual sums and actual estimated time.
            # Est Time = Total_Events * (60 / Total_Game_Events_Avg).
            # Let's assume 60 mins total TOI capacity.
            
            rates = {}
            def_rates = {}
            
            for st in target_states:
                ev_count = toi_counts[tid][st]
                xg = xg_sums.get((tid_float, st), 0.0)
                xga = xga_sums.get((tid_float, st), 0.0)
                
                # Rate per Event
                rate = xg / ev_count if ev_count > 0 else 0.0
                def_rate = xga / ev_count if ev_count > 0 else 0.0
                
                rates[st] = rate
                def_rates[st] = def_rate
                
            self.base_stats[tid] = {
                'rates': rates,
                'def_rates': def_rates,
                'toi_dist': dist,
                'total_events': total_ev,
                'pace': total_ev / 82.0 # events per game roughly? No, this is whole season.
                # Use League Avg Pace?
            }
            
        # Determine League Average Pace (Events per 60)
        # Total Events / Total Games / (60/60)
        n_games = df['game_id'].nunique()
        total_events_league = len(df)
        self.league_pace = (total_events_league / n_games) if n_games > 0 else 1.0  # Events per game (per 60)

    def predict(self, team_id: str, toi_dist: dict = None, pace: float = None):
        """
        Predicts GF/60 and GA/60.
        toi_dist: Optional override {'5v5': 0.8, ...}
        pace: Optional events per 60 override.
        """
        stats = self.base_stats.get(team_id)
        if not stats: return None
        
        # Get Proxies
        row = self.skill_proxies[self.skill_proxies['TeamID'] == team_id]
        if row.empty: return None
        
        # Defaults
        if toi_dist is None:
            # Normalize internal dist to sum to 1.0 over the 3 states?
            # Or keep raw?
            # Let's take the raw known states and normalize them to 1.0 for the prediction "pie"
            d = stats['toi_dist']
            total = d['5v5'] + d['5v4'] + d['4v5']
            if total == 0: total = 1
            toi_dist = {k: d[k]/total for k in ['5v5', '5v4', '4v5']}
            
        if pace is None:
            # Use Team's share of League Pace?
            # Or just League Pace (~150 events/game per team?)
            # self.league_pace is total events (both teams) per game. ~300.
            # TOI counts include all events witnessed (For + Against).
            # So we use the full League Pace to convert (xG/Event) * (Events/Game).
            pace = self.league_pace
            
        # Predict
        gf_60 = 0.0
        ga_60 = 0.0
        details = {}
        
        for st in ['5v5', '5v4', '4v5']:
            share = toi_dist.get(st, 0.0)
            base_xf = stats['rates'][st]
            base_xa = stats['def_rates'][st]
            
            # Skill
            mult_f = row.iloc[0].get(f'Offense_{st}', 1.0)
            mult_a = row.iloc[0].get(f'Defense_{st}', 1.0)
            
            # Calculation: (xG/Event) * Multiplier * (Events/60 * Share)
            pred_f = base_xf * mult_f * pace * share
            pred_a = base_xa * mult_a * pace * share
            
            gf_60 += pred_f
            ga_60 += pred_a
            
            details[st] = {
                'GF': pred_f, 'GA': pred_a, 
                'xGF_Base': base_xf * pace * share,
                'xGA_Base': base_xa * pace * share,
                'Mult_F': mult_f, 'Mult_A': mult_a
            }
            
        if team_id == '21': # COL
            logger.info(f"DEBUG PREDICT COL: Pace={pace:.1f}")
            for st in ['5v5', '5v4', '4v5']:
                share = toi_dist.get(st, 0.0)
                base_xf = stats['rates'][st]
                mult_f = row.iloc[0].get(f'Offense_{st}', 1.0)
                pred_f = base_xf * mult_f * pace * share
                logger.info(f"  {st}: BaseRate={base_xf:.4f}, Share={share:.3f}, Mult={mult_f:.3f} -> Pred={pred_f:.3f}")
                logger.info(f"       (xG Sum in Data: {stats['rates'][st] * stats['toi_dist'][st] * stats['total_events']:.1f})")

        return {'GF_60': gf_60, 'GA_60': ga_60, 'details': details}

def visualize_results(predictor, all_preds):
    """Generates verification plots."""
    df = pd.DataFrame(all_preds)
    
    # 1. Scatter: Predicted GF vs GA
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='GF_60', y='GA_60', hue='TeamAbbrev', palette='tab20', s=100)
    
    # Add labels
    for i, r in df.iterrows():
        plt.text(r['GF_60']+0.02, r['GA_60'], r['TeamAbbrev'], fontsize=9)
        
    plt.title(f'Predicted Performance (Skill Adjusted Goals/60) - {predictor.season}')
    plt.xlabel('Predicted Goals For / 60')
    plt.ylabel('Predicted Goals Against / 60')
    plt.gca().invert_yaxis() # Good defense is low
    plt.grid(True, alpha=0.3)
    
    out_path = Path('analysis/prediction_scatter.png')
    plt.savefig(out_path)
    logger.info(f"Saved scatter to {out_path}")
    
    # 2. Team Breakdown (Top 5 Offense)
    top_off = df.sort_values('GF_60', ascending=False).head(5)
    
    ids = top_off['TeamID'].tolist()
    # Collect components
    bfs = []
    
    for _, r in top_off.iterrows():
        d = r['details']
        for st in ['5v5', '5v4', '4v5']:
            bfs.append({
                'Team': r['TeamAbbrev'],
                'State': st,
                'GF': d[st]['GF']
            })
            
    bf_df = pd.DataFrame(bfs)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bf_df, x='Team', y='GF', hue='State')
    plt.title('Predicted GF Sources (Top 5 Offense Teams)')
    plt.ylabel('Goals Contribution / 60')
    
    out_path2 = Path('analysis/prediction_breakdown_top5.png')
    plt.savefig(out_path2)
    logger.info(f"Saved breakdown to {out_path2}")

def main():
    logger.info("Initializing Predictor...")
    # Initialize with refined loader
    pred = TeamPredictor()
    pred._load_data = pred._load_data_refined # Swap to correct method
    pred._load_data()
    
    results = []
    logger.info("Predicting for all teams...")
    
    for tid in pred.teams_list:
        # Need TeamID from Abbrev?
        # Predictor stores by TeamID.
        # Find ID for Abbrev
        row = pred.skill_proxies[pred.skill_proxies['TeamAbbrev'] == tid]
        if row.empty: continue
        real_id = str(row.iloc[0]['TeamID'])
        
        res = pred.predict(real_id)
        if res:
            res['TeamAbbrev'] = tid
            res['TeamID'] = real_id
            results.append(res)
            
    # Visualize
    visualize_results(pred, results)
    
    # Save CSV
    out_df = pd.DataFrame(results).drop(columns=['details'])
    out_df.to_csv('analysis/team_predictions.csv', index=False)
    logger.info("Predictions saved to analysis/team_predictions.csv")

if __name__ == "__main__":
    main()
