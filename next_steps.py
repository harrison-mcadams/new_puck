## This document outlines plans for next steps.

# General principles:
#   - Prioritize direct approaches. Previous iterations of the code have had
#   a lot of extra optionality to handle edge cases. For example, there is a
#   lot of code that handles different potential field names for a variable
#   of interest. While this is useful for robustness, it adds complexity. I
#   would like more direct control and assume a standard format.
#$   - Prioritize clarity over cleverness. The code should be easy to read
#   and understand, even if that means being more verbose or less optimized.

# Work on cacheing.
# Things to cache:
#   - website shortcuts: team list, game list, player list
#   - add remove shoot out attempts from game plotting

# Model improvements:
#   - add more features to xg model. next up shot type

# Refactoring:
#   - in parse.py (the routine that that works on dfs), create a routine
#   called filter. you should be able to filter the df on any number of
#   cases, but the obvious use case is 'team'. this is currently working in
#   analyze.py but should be generalized and moved to parse.py
#   - eventually rename plot.plot_events to plot._events

# Analysis
# The next concrete analysis task is to compute per-team mean xG maps for a season
# and compare them to the league-average xG map under the same filtering
# conditions (for example '5v5'). The goal: produce for each team a map of
# relative change vs league average (percentage change) and create a simple plot
# per team with helpful summary text.
#
# Below is a short specification, function contracts, pseudocode and notes to
# implement the feature in `analyze.py`.

# Specification (overview)
# - Input: season identifier (e.g. '20252026') or a season-level events DataFrame
#   that contains all events for the season. A `condition` dict controls which
#   rows to include (example: {'game_state': ['5v5'], 'is_net_empty': [0]}).
# - Output: for each team produce:
#     - team_mean_map: 2D grid of mean xG density for that team (or 'team' vs all 'others' if requested)
#     - league_map: 2D grid of mean xG density for league under same condition
#     - pct_change_map: (team_mean_map - league_map) / league_map * 100 (handle zeros)
#   and save or return a simple visualization (PNG) and a compact JSON summary
#   (total xG, xG/60 under that condition, integral of map, etc.)

# Function contracts (proposed functions to add to analyze.py)
# 1) compute_xg_heatmap_from_df(df, grid_res=1.0, sigma=6.0, x_col='x_a', y_col='y_a', amp_col='xgs', rink_mask_fn)
#    - Inputs:
#       df: events-level DataFrame (filtered for what we want: season/team/condition)
#       grid_res: grid spacing in feet (1.0 recommended)
#       sigma: gaussian kernel sigma in feet (6.0 recommended)
#       x_col, y_col: coordinate columns; default to adjusted coords 'x_a','y_a'
#       amp_col: column with amplitude per event (xG) default 'xgs'
#       rink_mask_fn: callable that given (XX,YY) returns boolean mask inside rink
#    - Outputs: (grid_x, grid_y, heatmap) where heatmap is 2D array (rows=ny, cols=nx)
#    - Behavior: builds uniform grid covering rink extents (-100..100, -42.5..42.5 by default),
#      accumulates gaussian kernels weighted by amp_col, masks outside rink using rink_mask_fn.
#
# 2) aggregate_maps_for_season(season_df, condition, by='team', grid_res=1.0, sigma=6.0)
#    - Inputs:
#       season_df: events-level DataFrame covering all games in season (must contain game_id, team_id/home_abb/away_abb, x_a,y_a,xgs, total_time_elapsed_seconds)
#       condition: dict to filter relevant rows (same format used elsewhere)
#       by: 'team' to compute per-team maps; other options possible later
#    - Outputs: league_map, dict(team -> {'map': team_map, 'total_xg':float, 'n_events':int, 'total_seconds':float})
#    - Behavior:
#       - Apply filter (reuse parse.build_mask or parse.filter once available)
#       - For league_map: for each game (or for whole season) compute heatmap of all events matching condition, then average per-game or normalize by total seconds (decide consistent approach). Recommended: sum xG across all matching events and divide by total observed seconds under condition -> xG per second heat density. Multiply by 3600 to get xG per hour (xG/3600), or by 60 for xG/60 per minute.
#       - For team_map: restrict events to team (team vs not-team logic); accumulate similarly and normalize by team total time under condition (or per-60 standardization).
#
# 3) compute_pct_change(team_map, league_map, eps=1e-9)
#    - elementwise (team - league) / (league + eps) * 100
#    - return pct_change map and a clipped version for plotting (e.g. cap at +/- 500%)
#
# 4) plot_team_vs_league_map(grid_x, grid_y, team_map, league_map, pct_map, out_path, title, cmap='RdBu')
#    - Simple plotting helper that draws rink (reuse rink.draw_rink) and overlays the pct_map with an opacity mapping similar to plot.plot_events heatmap style. Include a colorbar and a small summary text block (team, season, condition, total_xg/team, league_total_xg, xG/60 numbers).
#
# Pseudocode (high level)
# ------------------------
# def xg_maps_for_season(season, condition, grid_res=1.0, sigma=6.0, out_dir='static/season_maps'):
#     sdf = timing.load_season_df(season) or load events df before calling
#     df = sdf (must contain adjusted coords x_a/y_a and xgs)  # convert coordinates prior if needed
#     filtered = parse.build_mask(df, condition) -> df_cond = df[filtered]
#     league_map = compute_xg_heatmap_from_df(df_cond, grid_res, sigma)  # optionally normalize by total seconds
#     teams = unique team abbreviations in df (home_abb/away_abb)
#     for team in teams:
#         df_team = df_cond rows where event belongs to team (use existing helpers in plot.adjust_xy_for_homeaway or parse.filter)
#         team_map = compute_xg_heatmap_from_df(df_team, grid_res, sigma)
#         pct_map = compute_pct_change(team_map, league_map)
#         out_png = out_dir / f"{season}_{team}_pct_map.png"
#         plot_team_vs_league_map(...)
#
# Notes on normalization choices (explicit so implementer can pick):
# - Option A (recommended): produce heatmaps that represent xG per unit area (sum of xG in that cell). To compare between teams and league, normalize to xG per 60 minutes by dividing each map by the total seconds of data used and multiplying by 3600. This produces xG/60 spatial density.
# - Option B: compute per-game mean maps (map per game averaged across games). This reduces dominance of heavy-schedule teams but requires consistent per-game normalization.
# The code should pick one (A is simpler) and document it.
#
# Edge cases & safeguards
# - Cells in league_map can be zero; use small `eps` when computing percent change and document meaning when league_map is near zero.
# - Ensure df contains `xgs` floats. Rows missing xgs should be ignored (or assigned 0 if desired).
# - Ensure coordinates are adjusted (x_a/y_a) so all team shots align to expected side before aggregation. If x_a not present, call plot.adjust_xy_for_homeaway() or parse.adjust_xy logic earlier in pipeline.
# - If a team has very little data under the condition, skip or flag the map (e.g., less than N events or less than T seconds).
#
# Output and persistence
# - Save PNG plots into `static/league_vs_team_maps/{season}/{team}.png`.
# - Save a small JSON per team in same folder with summary stats: { 'team': 'PHI', 'season': '20252026', 'total_xg': x, 'total_seconds': s, 'xg_per60': v }
# - Optionally produce a CSV summary with one row per team with key metrics for quick reference.
#
# Example function usage (developer preview)
# -----------------------------------------
# from analyze import xg_maps_for_season
# xg_maps_for_season('20252026', condition={'game_state': ['5v5'], 'is_net_empty':[0]})
#
# Implementation hints
# - Reuse existing helpers: parse.build_mask (for filtering), plot.adjust_xy_for_homeaway (to create x_a/y_a), rink.rink_half_height_at_x (to mask cells outside rink).
# - Keep grid extents and resolution fixed across league/team maps so comparisons are direct.
# - Use numpy vectorized operations when building kernels for speed. If too slow, compute per-event contributions on a downsampled grid and refine later.
#
# Minimal deliverable for a first PR
# - Implement `compute_xg_heatmap_from_df` (clean, well-commented helper) in `analyze.py` or `plot.py`.
# - Implement `xg_maps_for_season` that:
#     1) loads/accepts season events df, applies `condition` filter,
#     2) computes league_map, then loops teams to compute team maps,
#     3) computes pct_change maps and saves PNG+JSON per team.
# - Add a small CLI wrapper in `analyze.py` `if __name__ == '__main__':` for quick runs.
#
# If you'd like I can implement the helper `compute_xg_heatmap_from_df` next and add a small demo in `analyze.py` that produces the league map and one team map for inspection.
#
# End of analysis plan.
