"""
team_assessment.py

Analyzes team performance by weighting Mixed Effects xG rates by actual time spent in each state.
Inputs: analysis/mixed_effects_heatmaps_20252026/team_stats_summary.json
Outputs: Stacked Bar Chart of Expected xG per Typical Game
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    season = "20252026"
    in_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
    json_path = in_dir / "team_stats_summary.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded stats for {len(data)} teams.")
    
    # States to consider
    states = ['5v5', '5v4', '4v5']
    
    rows = []
    
    for team, buckets in data.items():
        # Get games played (max across buckets)
        games_played = 0
        for s in buckets.values():
            if s['games_played'] > games_played:
                games_played = s['games_played']
                
        if games_played == 0: continue
        
        row = {'Team': team, 'Games': games_played}
        
        total_xg_for_typ = 0.0
        total_xg_against_typ = 0.0
        raw_minutes = {}
        
        for state in states:
            if state not in buckets:
                # Fill zeros
                row[f'Min_{state}'] = 0.0
                row[f'GF60_{state}'] = 0.0
                row[f'GA60_{state}'] = 0.0
                row[f'ExpGF_{state}'] = 0.0
                row[f'ExpGA_{state}'] = 0.0
                continue
                
            s = buckets[state]
            seconds = s['seconds']
            # Avoid div by zero
            if seconds < 60:
                xgf60 = 0
                xga60 = 0
            else:
                xgf60 = (s['xg_for'] / seconds) * 3600
                xga60 = (s['xg_against'] / seconds) * 3600
                
            avg_min = (seconds / 60) / games_played
            row[f'Min_Raw_{state}'] = avg_min
            raw_minutes[state] = avg_min
            
            # Store rates for next step
            row[f'GF60_{state}'] = xgf60
            row[f'GA60_{state}'] = xga60
            
        # Scaling to 60 minutes
        total_raw_min = sum(raw_minutes.values())
        if total_raw_min > 0:
            scale_factor = 60.0 / total_raw_min
        else:
            scale_factor = 1.0
            
        row['Scale_Factor'] = scale_factor
        
        for state in states:
            if state in raw_minutes:
                raw_m = raw_minutes[state]
                scaled_m = raw_m * scale_factor
                
                xgf60 = row[f'GF60_{state}']
                xga60 = row[f'GA60_{state}']
                
                exp_gf = xgf60 * (scaled_m / 60)
                exp_ga = xga60 * (scaled_m / 60)
                
                row[f'Min_{state}'] = scaled_m
                row[f'ExpGF_{state}'] = exp_gf
                row[f'ExpGA_{state}'] = exp_ga
                
                total_xg_for_typ += exp_gf
                total_xg_against_typ += exp_ga
            
        row['Total_ExpGF'] = total_xg_for_typ
        row['Total_ExpGA'] = total_xg_against_typ
        row['Net_ExpG'] = total_xg_for_typ - total_xg_against_typ
        
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df = df.sort_values('Net_ExpG', ascending=False)
    
    # Save CSV
    df.to_csv(in_dir / "team_assessment.csv", index=False)
    print(f"Saved assessment data to {in_dir / 'team_assessment.csv'}")
    
    # PLOT 1: Stacked Bar Chart (For and Against)
    # Fig with 2 subplots: For and Against
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    teams = df['Team']
    x = np.arange(len(teams))
    width = 0.8
    
    # Components
    # 5v5, 5v4 (PP), 4v5 (PK - usually low GF, high GA)
    
    # Plot xGF components
    p1 = axes[0].bar(x, df['ExpGF_5v5'], width, label='5v5', color='#1f77b4')
    p2 = axes[0].bar(x, df['ExpGF_5v4'], width, bottom=df['ExpGF_5v5'], label='5v4 (PP)', color='#ff7f0e')
    p3 = axes[0].bar(x, df['ExpGF_4v5'], width, bottom=df['ExpGF_5v5']+df['ExpGF_5v4'], label='4v5 (PK)', color='#2ca02c')
    
    axes[0].set_ylabel('Expected Goals For (per Typical Game)')
    axes[0].set_title('Offensive Contribution by Game State')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot xGA components
    p4 = axes[1].bar(x, df['ExpGA_5v5'], width, label='5v5', color='#1f77b4')
    p5 = axes[1].bar(x, df['ExpGA_5v4'], width, bottom=df['ExpGA_5v5'], label='5v4 (PP)', color='#ff7f0e') # Usually low
    p6 = axes[1].bar(x, df['ExpGA_4v5'], width, bottom=df['ExpGA_5v5']+df['ExpGA_5v4'], label='4v5 (PK)', color='#d62728') # PK GA
    
    axes[1].set_ylabel('Expected Goals Against (per Typical Game)')
    axes[1].set_title('Defensive Liability by Game State')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.xticks(x, teams, rotation=45)
    plt.tight_layout()
    plt.savefig(in_dir / "team_assessment_components.png", dpi=120)
    plt.close()
    
    # PLOT 2: Scatter of Net Rating (Total xGF vs Total xGA)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df['Total_ExpGA'], df['Total_ExpGF'], s=100, alpha=0.8)
    
    for i, txt in enumerate(df['Team']):
        ax.annotate(txt, (df['Total_ExpGA'].iloc[i], df['Total_ExpGF'].iloc[i]), xytext=(5,5), textcoords='offset points')
        
    # Draw diagonals
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Even')
    
    ax.set_xlabel('Total Expected Goals Against')
    ax.set_ylabel('Total Expected Goals For')
    ax.set_title('Team Assessment: Weighted Performance (Typical Game)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(in_dir / "team_assessment_scatter.png", dpi=120)
    plt.close()
    
    print("Done. Generated charts.")

if __name__ == "__main__":
    main()
