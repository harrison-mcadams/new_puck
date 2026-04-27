import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from . import config
from . import analyze
from . import plot

logger = logging.getLogger(__name__)

def parse_series_id(game_id: Any) -> str:
    """
    Extract Series ID from Playoff Game ID.
    Game ID format: YYYY030WSR
    Round W (7th digit), Series S (8th digit)
    """
    gid_str = str(game_id)
    if len(gid_str) < 10 or gid_str[4:6] != '03':
        return "unknown"
    
    # 2025030161 -> Series ID: 202503016
    return gid_str[:9]

def group_playoff_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group playoff games into series.
    Returns a dictionary mapping series_id to its events DataFrame.
    """
    if df.empty:
        return {}
    
    # Filter for playoff games (Type 03)
    df = df[df['game_id'].astype(str).str[4:6] == '03'].copy()
    if df.empty:
        return {}
    
    df['series_id'] = df['game_id'].apply(parse_series_id)
    series_groups = {}
    for sid, group in df.groupby('series_id'):
        series_groups[sid] = group
    
    return series_groups

def get_series_metadata(series_id: str, df_series: pd.DataFrame) -> Dict[str, Any]:
    """
    Return metadata for a series (Teams, Games, etc.)
    """
    game_ids = sorted(df_series['game_id'].unique().tolist())
    
    # Identify teams (home/away from the first game)
    first_game = df_series[df_series['game_id'] == game_ids[0]]
    home_abb = first_game['home_abb'].dropna().unique()[0]
    away_abb = first_game['away_abb'].dropna().unique()[0]
    
    # Extract Round and Series Number
    round_num = int(series_id[7:8])
    series_num = int(series_id[8:9])
    
    return {
        'series_id': series_id,
        'round': round_num,
        'series_number': series_num,
        'home_team': home_abb,
        'away_team': away_abb,
        'game_ids': game_ids,
        'num_games': len(game_ids)
    }

def generate_playoff_plots(season: str = '20252026', force: bool = False):
    """
    Main entry point to generate all playoff-related plots.
    """
    target_season = f"{season}_playoffs"
    csv_path = os.path.join(config.DATA_DIR, f"{target_season}.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Playoff data not found: {csv_path}")
        return
    
    logger.info(f"Loading playoff data from {csv_path}...")
    df_playoffs = pd.read_csv(csv_path)
    
    # Ensure xGs are present
    df_playoffs, _, _ = analyze._predict_xgs(df_playoffs)
    
    series_groups = group_playoff_series(df_playoffs)
    logger.info(f"Found {len(series_groups)} playoff series.")
    
    playoffs_analysis_dir = os.path.join(config.ANALYSIS_DIR, 'playoffs', season)
    os.makedirs(playoffs_analysis_dir, exist_ok=True)
    
    series_list = []
    
    for sid, df_series in series_groups.items():
        meta = get_series_metadata(sid, df_series)
        series_dir = os.path.join(playoffs_analysis_dir, sid)
        os.makedirs(series_dir, exist_ok=True)
        
        logger.info(f"Processing Series {sid}: {meta['home_team']} vs {meta['away_team']}")
        
        # 1. Individual Game Plots
        game_metadata = []
        for gid in meta['game_ids']:
            game_dir = os.path.join(series_dir, str(gid))
            os.makedirs(game_dir, exist_ok=True)
            
            df_game = df_series[df_series['game_id'] == gid]
            
            # Heatmap
            heatmap_path = os.path.join(game_dir, 'heatmap.png')
            if force or not os.path.exists(heatmap_path):
                try:
                    analyze.xgs_map(
                        data_df=df_game,
                        condition={},
                        out_path=heatmap_path,
                        show=False,
                        return_heatmaps=False,
                        events_to_plot=['shot-on-goal', 'goal', 'xgs'],
                        heatmap_split_mode='team_not_team',
                        team_for_heatmap=meta['home_team']
                    )
                except Exception as e:
                    logger.error(f"Failed game heatmap {gid}: {e}")
            
            # Game Worm
            worm_path = os.path.join(game_dir, 'worm.png')
            if force or not os.path.exists(worm_path):
                try:
                    plot.plot_game_worm(df_game, worm_path, team_for_heatmap=meta['home_team'])
                except Exception as e:
                    logger.error(f"Failed game worm {gid}: {e}")
            
            game_metadata.append({
                'game_id': int(gid),
                'heatmap': f"playoffs/{season}/{sid}/{gid}/heatmap.png",
                'worm': f"playoffs/{season}/{sid}/{gid}/worm.png"
            })
            
        # 2. Aggregate Series Plots
        agg_heatmap_path = os.path.join(series_dir, 'aggregate_heatmap.png')
        if force or not os.path.exists(agg_heatmap_path):
            try:
                # For aggregate, we use 'team_not_team' mode targeting the home team
                # to show Home vs Away overall.
                analyze.xgs_map(
                    data_df=df_series,
                    condition={},
                    out_path=agg_heatmap_path,
                    show=False,
                    return_heatmaps=False,
                    events_to_plot=['shot-on-goal', 'goal', 'xgs'],
                    heatmap_split_mode='team_not_team',
                    team_for_heatmap=meta['home_team']
                )
            except Exception as e:
                logger.error(f"Failed aggregate heatmap {sid}: {e}")
                
        agg_worm_path = os.path.join(series_dir, 'aggregate_worm.png')
        if force or not os.path.exists(agg_worm_path):
            try:
                # plot_game_worm handles multiple games if they are in the DF, 
                # but it might need adjustment to show them sequentially or aggregated.
                # Currently it plots by time. For aggregate series, maybe a simple cumulative?
                # The current plot_game_worm is designed for a single game.
                # Let's use a specialized series worm or just pass the whole DF.
                plot.plot_game_worm(df_series, agg_worm_path, team_for_heatmap=meta['home_team'])
            except Exception as e:
                logger.error(f"Failed aggregate worm {sid}: {e}")
        
        meta['aggregate_heatmap'] = f"playoffs/{season}/{sid}/aggregate_heatmap.png"
        meta['aggregate_worm'] = f"playoffs/{season}/{sid}/aggregate_worm.png"
        meta['games'] = game_metadata
        series_list.append(meta)
        
    # Save series summary for the web UI
    import json
    with open(os.path.join(playoffs_analysis_dir, 'series_summary.json'), 'w') as f:
        json.dump(series_list, f, indent=2)
    
    return series_list
