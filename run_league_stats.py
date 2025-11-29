import analyze
import os
import json
import pandas as pd

if __name__ == '__main__':
    season = '20252026'
    # The output directory will be created by analyze.season_analysis inside the passed out_dir
    # But analyze.season_analysis appends /{season} to the out_dir.
    # So if I pass 'static', it saves to 'static/20252026'.
    # If I pass 'static/league_stats', it saves to 'static/league_stats/20252026'.
    # app.py expects `static/20252026_season_analysis/20252026_team_summary.json`.
    # So I should pass `out_dir='static/20252026_season_analysis'`?
    # No, wait. If analyze.season_analysis appends season, then passing `out_dir='static'` results in `static/20252026`.
    # I'll pass `out_dir='static/league_stats'` and adjust app.py to match.
    # Or I can just manually save the summary where app.py expects it.

    # Let's use a dedicated directory for this run to avoid clutter.
    base_out_dir = 'static/league_stats'
    os.makedirs(base_out_dir, exist_ok=True)

    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}

    print(f"Running season analysis for {season} with condition {condition}...")
    # analyze.season_analysis signature:
    # def season_analysis(season=..., out_dir=..., ...)
    # It saves to out_dir/{season}/...
    results = analyze.season_analysis(
        season=season,
        condition=condition,
        out_dir=base_out_dir
    )

    # The files are now in static/league_stats/20252026/

    summary_list = []
    # season_analysis returns {'league_map': ..., 'teams': {...}, ...}
    team_results = results.get('teams', {})
    for team, data in team_results.items():
        if team == 'league_map': continue
        if 'error' in data: continue
        
        stats = data.get('summary_stats', {})
        
        # Image is at static/league_stats/{team}_xg_map.png
        img_path = f"league_stats/{team}_xg_map.png"
        
        gf = int(stats.get('team_goals', 0))
        ga = int(stats.get('opp_goals', 0))
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
            'Image': img_path
        }
        summary_list.append(row)

    # Save consolidated summary to the same directory
    summary_path = f'{base_out_dir}/summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_list, f, indent=2)

    print(f"Saved summary to {summary_path}")
