# Relative Heatmap Refinement Summary

## Overview
Refined the visual presentation and interpretability of 5v5 relative xG heatmaps per user feedback. Addressed visual artifacts, scaling, and missing plot types.

## Key Visual Changes (Final)
*   **Rink Boundary**: 
    *   **Fixed Discontinuity**: Implemented explicit drawing of straight lines and `patches.Arc` curves with `linewidth=1.2` and `clip_on=False`.
    *   **Background**: Unified to pure white (`#ffffff`).
*   **Scaling**: 
    *   Values scaled **100x** (per 100 sq ft).
    *   `vmax=0.02` for 5v5 and Special Teams.

## Restored & Added Features
*   **Special Teams (Combined)**: 
    *   **Visual Parity**: Reimplemented plotting to use `plot_relative_map`, ensuring identical colorbars (`vmax=0.02`), summary text, and percentile calculations as 5v5 plots.
    *   **Methodology**: Uses difference from league average (subtraction) consistent with 5v5.
    *   **Workflow**: Integrated into `run_league_stats.py` with proper `.npy` map persistence.
*   **Player Scatter Plots**:
    *   **League-Wide**: `analysis/players/{season}/scatter_players_league.png`
        *   **Filter**: > 5 games played.
        *   **Labels**: Only outliers (> 2.0 std dev) labeled.
        *   **Reference**: Added Unity Line (x=y).
        *   **Average**: Uses **League Weighted Mean** (Total xG / Total Seconds) for symmetric avg lines.
    *   **Per-Team**: `analysis/players/{season}/scatter_players_{TEAM}.png`
        *   **Labels**: All players labeled.
    *   **Enhancement**: Fixed team inference logic.
*   **Team Scatter Plot**:
    *   Added **League Average** red dotted lines to `analysis/league/{season}/{cond}/scatter.png`.

## Files Modified
*   `puck/rink.py`: Boundary drawing logic.
*   `puck/analyze.py`: 
    *   `generate_player_scatter_plots`: Filters, Weighted Mean, Unity Line.
    *   `generate_special_teams_plot`: Refactored to use `plot_relative_map`.
    *   `generate_scatter_plot`: Avg lines.
*   `scripts/run_league_stats.py`: Integrated ST generation, restored `.npy` saving.
*   `scripts/daily.py`: Removed restrictions.

## Current Status
*   **Completed**: All artifacts generated and verified. `daily.py` pipeline is fully updated.
