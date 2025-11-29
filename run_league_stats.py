import analyze
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

def generate_scatter_plot(summary_list, out_dir):
    """Generate xGF/60 vs xGA/60 scatter plot."""
    if not summary_list:
        return

    df = pd.DataFrame(summary_list)
    
    # Check if required columns exist
    if 'xGF/60' not in df.columns or 'xGA/60' not in df.columns:
        print("Missing xGF/60 or xGA/60 columns for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(df['xGF/60'], df['xGA/60'], alpha=0.7)
    
    # Add labels
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['xGF/60'], row['xGA/60']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Invert Y axis (lower xGA is better, usually top-right is good/good, but standard is top-right is high-event)
    # Usually: x-axis = xGF/60 (right is better), y-axis = xGA/60 (lower is better, so invert?)
    # Let's stick to standard math axes for now, but maybe invert Y so top is "Good Defense"?
    # For now, standard axes. High xGA is bad (top), Low xGA is good (bottom).
    # High xGF is good (right), Low xGF is bad (left).
    # So Bottom-Right is Best.
    
    ax.set_xlabel('xGF/60')
    ax.set_ylabel('xGA/60')
    ax.set_title('Team xG Rates (xGF/60 vs xGA/60)')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add mean lines
    mean_xgf = df['xGF/60'].mean()
    mean_xga = df['xGA/60'].mean()
    ax.axvline(mean_xgf, color='k', linestyle=':', alpha=0.5, label=f'Avg xGF: {mean_xgf:.2f}')
    ax.axhline(mean_xga, color='k', linestyle=':', alpha=0.5, label=f'Avg xGA: {mean_xga:.2f}')
    ax.legend()
    
    # Save
    out_path = os.path.join(out_dir, 'scatter.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter plot to {out_path}")

if __name__ == '__main__':
    season = '20252026'
    game_states = ['5v5', '5v4', '4v5']
    
    for state in game_states:
        print(f"\nProcessing Game State: {state}")
        
        # Define output directory: static/league_stats/{season}/{state}
        # Note: analyze.season_analysis appends {season} to out_dir if we are not careful?
        # analyze.season_analysis:
        #   if out_dir is None: out_dir = os.path.join('static', f'{season}_season_analysis')
        #   os.makedirs(out_dir, exist_ok=True)
        #   It does NOT append season if out_dir is provided.
        #   Wait, let's check analyze.py line 550.
        #   It uses the provided out_dir directly.
        
        # We want: static/league_stats/20252026/5v5/
        base_out_dir = os.path.join('static', 'league_stats', season, state)
        os.makedirs(base_out_dir, exist_ok=True)

        condition = {'game_state': [state], 'is_net_empty': [0]}

        print(f"Running season analysis for {season} with condition {condition}...")
        
        try:
            results = analyze.season_analysis(
                season=season,
                condition=condition,
                out_dir=base_out_dir
            )
        except Exception as e:
            print(f"Error running season analysis for {state}: {e}")
            continue

        summary_list = []
        team_results = results.get('teams', {})
        
        for team, data in team_results.items():
            if team == 'league_map': continue
            if 'error' in data: continue
            
            stats = data.get('summary_stats', {})
            
            # Image is at {base_out_dir}/{team}_xg_map.png
            # We need the relative path for the frontend
            # Frontend expects: static/...
            # So we store: league_stats/{season}/{state}/{team}_xg_map.png
            img_path = f"league_stats/{season}/{state}/{team}_xg_map.png"
            
            # Fix GA key: analyze.py produces 'other_goals', not 'opp_goals'
            gf = int(stats.get('team_goals', 0))
            ga = int(stats.get('other_goals', 0)) # FIXED: was 'opp_goals'
            xgf = float(stats.get('team_xgs', 0))
            xga = float(stats.get('other_xgs', 0))
            cf = int(stats.get('team_attempts', 0))
            ca = int(stats.get('opp_attempts', 0))
            
            # Calculate percentages
            cf_pct = round(cf / (cf + ca) * 100, 1) if (cf + ca) > 0 else 0.0
            xgf_pct = round(xgf / (xgf + xga) * 100, 1) if (xgf + xga) > 0 else 0.0
            gf_pct = round(gf / (gf + ga) * 100, 1) if (gf + ga) > 0 else 0.0
            
            row = {
                'Team': team,
                'GP': stats.get('n_games', 0),
                'TOI': round(stats.get('team_seconds', 0) / 60, 1), # Minutes
                'GF': gf,
                'GA': ga,
                'GF%': gf_pct,
                'xGF': round(xgf, 2),
                'xGA': round(xga, 2),
                'xGF%': xgf_pct,
                'CF': cf,
                'CA': ca,
                'CF%': cf_pct,
                'xGF/60': round(stats.get('team_xg_per60', 0), 2),
                'xGA/60': round(stats.get('other_xg_per60', 0), 2),
                'Image': img_path,
                'RelativeMap': f"league_stats/{season}/{state}/{team}_relative_map.png"
            }
            summary_list.append(row)

        # Save consolidated summary
        summary_path = os.path.join(base_out_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_list, f, indent=2)

        print(f"Saved summary to {summary_path}")
        
        # Generate Scatter Plot
        generate_scatter_plot(summary_list, base_out_dir)

