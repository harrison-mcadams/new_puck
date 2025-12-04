import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import timing
import analyze
import nhl_api

def analyze_team_skill(season='20252026', skill_weight=0.5):
    """
    Analyze Team Skill (Goals Above Expected) for Offense and Defense.
    
    Args:
        season (str): Season string.
        skill_weight (float): Regression factor (0.0 = All Luck/Mean, 1.0 = All Skill).
                              Lower values regress heavily to the mean (1.0).
    """
    print(f"--- Analyzing Team Skill (GAx) for {season} ---")
    
    # 1. Load Data
    print("Loading season data...")
    df = timing.load_season_df(season)
    if df is None or df.empty:
        print("No data found.")
        return

    # Ensure xGs
    print("Ensuring xG predictions...")
    df, _, _ = analyze._predict_xgs(df)
    
    # 2. Aggregate Stats by Team
    # We need to attribute Goals and xG to the correct team.
    # For each event:
    #   - Event Team (Offense): Goals For, xG For
    #   - Opponent Team (Defense): Goals Against, xG Against
    
    print("Aggregating team stats...")
    
    # Identify teams
    teams = sorted(df['home_abb'].dropna().unique())
    
    team_stats = {}
    
    for team in teams:
        # Get team ID
        # (Simplified: assume we can filter by home_abb/away_abb)
        # Filter games involving team
        mask = (df['home_abb'] == team) | (df['away_abb'] == team)
        df_team = df[mask]
        
        # We need to be careful about "For" and "Against"
        # Let's iterate or use vectorized operations?
        # Vectorized is better.
        
        # Events where Team is the Event Team (Offense)
        # We need team_id.
        # Let's get team_id from the first game
        try:
            sample = df[df['home_abb'] == team]
            if not sample.empty:
                tid = sample.iloc[0]['home_id']
            else:
                sample = df[df['away_abb'] == team]
                tid = sample.iloc[0]['away_id']
        except:
            continue
            
        # Offense: Events where team_id == tid
        # Defense: Events where team_id != tid (and game involves tid)
        
        # Filter df_team again to be sure
        # (df_team contains all events from games involving team)
        
        # Offense
        off_mask = (df_team['team_id'] == tid)
        gf = df_team[off_mask & (df_team['event'] == 'goal')].shape[0]
        xgf = df_team[off_mask]['xgs'].sum()
        
        # Defense
        def_mask = (df_team['team_id'] != tid)
        ga = df_team[def_mask & (df_team['event'] == 'goal')].shape[0]
        xga = df_team[def_mask]['xgs'].sum()
        
        team_stats[team] = {
            'gf': gf,
            'xgf': xgf,
            'ga': ga,
            'xga': xga,
            'games': df_team['game_id'].nunique()
        }
        
    # 3. Calculate Factors
    print("Calculating skill factors...")
    results = []
    
    for team, stats in team_stats.items():
        if stats['xgf'] <= 0 or stats['xga'] <= 0:
            continue
            
        # Raw Factors
        raw_off_factor = stats['gf'] / stats['xgf']
        raw_def_factor = stats['ga'] / stats['xga'] # Lower is better for defense (allowing fewer goals than xG)
        
        # Regress to Mean (1.0)
        # Factor = 1.0 + (Raw - 1.0) * weight
        off_factor = 1.0 + (raw_off_factor - 1.0) * skill_weight
        def_factor = 1.0 + (raw_def_factor - 1.0) * skill_weight
        
        # GAx (Goals Above Expected) - Total
        gax_off = stats['gf'] - stats['xgf']
        gax_def = stats['xga'] - stats['ga'] # Positive means "Saved Goals Above Expected"
        
        results.append({
            'team': team,
            'off_factor': off_factor,
            'def_factor': def_factor,
            'raw_off': raw_off_factor,
            'raw_def': raw_def_factor,
            'gax_off': gax_off,
            'gax_def': gax_def,
            'games': stats['games']
        })
        
    df_res = pd.DataFrame(results)
    
    # 4. Visualization
    print("Generating plots...")
    out_dir = f"data/{season}/skill"
    os.makedirs(out_dir, exist_ok=True)
    
    # Scatter Plot: Offense vs Defense Factors
    plt.figure(figsize=(10, 10))
    
    # X: Offense Factor (Higher is Better)
    # Y: Defense Factor (Lower is Better - fewer goals allowed)
    # Let's invert Y axis so "Top Right" is "Best" (High Off, Low Def Factor)
    
    x = df_res['off_factor']
    y = df_res['def_factor']
    teams = df_res['team']
    
    plt.scatter(x, y, c='blue', alpha=0.6)
    
    # Add labels
    for i, txt in enumerate(teams):
        plt.annotate(txt, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    # Reference Lines
    plt.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Team Skill Factors (Regressed w={skill_weight})\nOffense vs Defense (Goals/xG)", fontsize=14, fontweight='bold')
    plt.xlabel("Offense Factor (Goals For / xG For)\n> 1.0 = Good Finishing / Luck", fontsize=12)
    plt.ylabel("Defense Factor (Goals Against / xG Against)\n< 1.0 = Good Goaltending / Luck", fontsize=12)
    
    # Invert Y axis so "Good Defense" is up? 
    # Or just label quadrants.
    # Let's invert Y so Top-Right is Best.
    plt.gca().invert_yaxis()
    
    # Quadrant Labels
    # Top Right (High Off, Low Def Factor) -> Contenders
    plt.text(1.1, 0.9, "CONTENDERS\n(Good Finish, Good Saves)", ha='center', va='center', color='green', fontweight='bold', alpha=0.5)
    # Bottom Left (Low Off, High Def Factor) -> Rebuilders
    plt.text(0.9, 1.1, "REBUILDERS\n(Bad Finish, Bad Saves)", ha='center', va='center', color='red', fontweight='bold', alpha=0.5)
    # Top Left (Low Off, Low Def Factor) -> Boring / Defensive
    plt.text(0.9, 0.9, "DEFENSIVE\n(Bad Finish, Good Saves)", ha='center', va='center', color='orange', fontweight='bold', alpha=0.5)
    # Bottom Right (High Off, High Def Factor) -> Fun / All Gas No Brakes
    plt.text(1.1, 1.1, "FUN\n(Good Finish, Bad Saves)", ha='center', va='center', color='purple', fontweight='bold', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/team_skill_scatter.png", dpi=150)
    plt.close()
    
    # Bar Charts (Top 5 / Bottom 5)
    # Finishing
    df_sorted_off = df_res.sort_values('off_factor', ascending=False)
    top_off = df_sorted_off.head(5)
    bot_off = df_sorted_off.tail(5)
    
    # Goaltending (Defense Factor - Lower is Better)
    df_sorted_def = df_res.sort_values('def_factor', ascending=True)
    top_def = df_sorted_def.head(5) # Lowest factors
    bot_def = df_sorted_def.tail(5) # Highest factors
    
    print("\n--- Top 5 Finishing Teams ---")
    print(top_off[['team', 'off_factor', 'raw_off', 'gax_off']])
    
    print("\n--- Top 5 Goaltending Teams ---")
    print(top_def[['team', 'def_factor', 'raw_def', 'gax_def']])
    
    # Save Summary CSV
    df_res.to_csv(f"{out_dir}/team_skill_summary.csv", index=False)
    print(f"\nAnalysis complete. Results saved to {out_dir}")

if __name__ == "__main__":
    analyze_team_skill()
