"""data_pipeline.py

Centralized pipeline for preprocessing PBP data for Training and Inference.
Refactored from logic previously in scripts/train_xgboost_model.py and puck/analyze.py.

ARCHITECTURE NOTE - COORDINATE FLOW:
1. Raw API Data (PBP) arrives with 'blocked-shots' already attributed to the SHOOTER's team.
2. correction.fix_blocked_shot_attribution(): 
   - NO-OP (kept for backward compatibility and pipeline structure).
   - Historically swapped team_id, but API verification (2026-05-03) confirmed this is unnecessary.
3. Orientation Standardization:
   - ALL shots are flipped to a "Right-Attack" orientation (x towards +89 goal).
   - This is the canonical frame for ALL models (XGBoost, GLM, etc.).
4. Synchronization:
   - 'x/y' and 'x_adj/y_adj' MUST be flipped in tandem. 
   - Coordinate flipping is IDEMPOTENT; multiple runs detect attacking_side to avoid double-flipping.
"""

import pandas as pd
import numpy as np
import warnings
import logging
from typing import Optional, List, Tuple

from . import correction, impute, arena_adjustments, features, config, html_enrichment

NUMERIC_DEFAULTS = {
    'is_rebound': 0,
    'rebound_angle_change': 0.0,
    'rebound_time_diff': 0.0,
    'rebound_source': 'none',
    'is_rush': 0,
    'last_event_time_diff': 2.0,
    'score_diff': 0,
    'distance': -1.0, 
    'angle_deg': 0.0,
    'time_elapsed_in_period_s': 0.0,
    'total_time_elapsed_s': 0.0,
    'dist_from_last_event': 15.0,
    'speed_from_last_event': 7.5
}

def preprocess_features(df_input: pd.DataFrame, 
                        is_training: bool = False, 
                        verbose: bool = False,
                        apply_arena_adjustments: bool = True,
                        apply_imputation: bool = True,
                        apply_dithering: bool = False,
                        apply_filtering: bool = False,
                        apply_bio_enrichment: bool = True,
                        apply_html_enrichment: bool = True,
                        apply_attribution_fix: bool = True,
                        impute_alpha: float = 0.2,
                        exclude_blocked: bool = False,
                        game_id: Optional[str] = None) -> pd.DataFrame:
    """Preprocess NHL game data for model consumption.
    
    This pipeline handles coordinate normalization, team orientation, 
    and critical data corrections.
    
    NOTE ON BLOCKED SHOTS:
    The NHL API attributes a 'blocked-shot' event to the team of the SHOOTER (the attacking team). 
    This was confirmed via a comprehensive audit across all modern-era seasons. 
    
    PROCESSING STEPS:
    1. Team Attribution: Already correct in source data.
    2. Orientation: Standardizing coordinates to the attacker's 'scoring' orientation.
    3. Shot Type: Recovering 'shot_type' via fuzzy-matching with NHL HTML Play-by-Play reports.
    """
    
    # Work on copy
    df = df_input.copy()
    
    if len(df) == 0:
        return df

    def vprint(*args):
        if verbose:
            print(*args)

    vprint(f"Preprocessing {len(df)} rows. Training={is_training}, Impute={apply_imputation}, Adjust={apply_arena_adjustments}")
    
    # 0. Ensure coordinates are numeric early to avoid dithering/correction errors
    for col in ['x', 'y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # Standardize Event Names
    if 'event' in df.columns:
        # 1. Step: Fix blocked shot attribution
        # NOTE: As of 2026-05-03, this is a NO-OP because the API already uses shooter attribution.
        if apply_attribution_fix and (df['event'].astype(str).str.lower() == 'blocked-shot').any():
            vprint("  Attribution Check: Blocked Shots (API already uses Shooter)...")
            df = correction.fix_blocked_shot_attribution(df)

        # 2. Step: Standardize tokens
        s_event = df['event'].astype(str).str.lower()
        evt_map = {
            'shot': 'shot-on-goal',
            'shots': 'shot-on-goal',
            'missed shot': 'missed-shot',
            'blocked shot': 'blocked-shot',
            'goal': 'goal'
        }
        # Use Series fillna for type safety
        s_mapped = pd.Series(s_event).map(evt_map)
        df['event'] = s_mapped.fillna(pd.Series(s_event)).values
        
    if exclude_blocked and 'event' in df.columns:
        n_blocked = (df['event'] == 'blocked-shot').sum()
        if n_blocked > 0:
            vprint(f"  Filtering out {n_blocked} blocked shots per exclude_blocked config...")
            df = df[df['event'] != 'blocked-shot'].copy()

    # Derive is_home early 
    if 'team_id' in df.columns and 'home_id' in df.columns:
        # Strip decimal points for robust comparison (e.g. '16.0' -> '16')
        tid_s = df['team_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        hid_s = df['home_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['is_home'] = (tid_s == hid_s).astype(int)

    # Sync x/y with x_adj/y_adj BEFORE standardization if they exist
    # This prevents standardization from only affecting one set of coordinates
    if 'x_adj' in df.columns:
        df['x'] = df['x_adj'].copy()
    if 'y_adj' in df.columns:
        df['y'] = df['y_adj'].copy()

    # 1.5 Parse Relative Game State
    if 'game_state' in df.columns and 'is_home' in df.columns:
        # Use str.extract for more robustness than split
        extracted = df['game_state'].astype(str).str.extract(r'(\d+)v(\d+)')
        if not extracted.isna().all().all():
            home_count = extracted[0].fillna('5')
            away_count = extracted[1].fillna('5')
            is_home_mask = (df['is_home'] == 1)
            
            # Relative state is Offense v Defense
            # If shooter is home, it's Home v Away (GS)
            # If shooter is away, it's Away v Home (GS swapped)
            df['relative_game_state'] = np.where(
                is_home_mask,
                df['game_state'].astype(str),
                away_count + 'v' + home_count
            )
        else:
            df['relative_game_state'] = df['game_state'].astype(str)
    elif 'game_state' in df.columns:
        df['relative_game_state'] = df['game_state'].astype(str)

    # 1.6 HTML Enrichment
    if apply_html_enrichment:
        if game_id:
            vprint(f"  Enriching shots with HTML PBP for game {game_id}...")
            df = html_enrichment.enrich_blocks_with_html(df, game_id)
        elif 'game_id' in df.columns:
            # Handle multi-game enrichment
            unique_games = df['game_id'].unique()
            if len(unique_games) > 1:
                from joblib import Parallel, delayed
                vprint(f'  Enriching shots with HTML PBP for {len(unique_games)} games in parallel (optimized)...')
                game_groups = [(str(g_id), group) for g_id, group in df.groupby('game_id')]
                def process_group(args):
                    g_id_str, sub_df = args
                    return html_enrichment.enrich_blocks_with_html(sub_df, g_id_str)
                enriched_frames = Parallel(n_jobs=-1)(delayed(process_group)(arg) for arg in game_groups)
                df = pd.concat(enriched_frames, ignore_index=True)

    # 2. Standardize Orientation (Canonical "Right-Attack" Frame)
    # We rotate/flip all shots so that the shooting team is attacking the goal at x=89.0.
    # This normalization is required for all models to learn consistent spatial patterns.
    # 
    # IDEMPOTENCY: We determine the "attacking side" based on current home_defending_side 
    # and the team_id. If a shot is already flipped, the calculation below will identify 
    # it as "Right-Attack" and omit the second flip.
    if 'x' in df.columns and 'home_team_defending_side' in df.columns and 'team_id' in df.columns and 'home_id' in df.columns:
        # Determine Coordinate Flip
        side_str = df['home_team_defending_side'].astype(str).str.lower().str.strip()
        
        # semantics: 'left' means Home defends Left (-89) and attacks Right (+89).
        def_side_sign = side_str.map({'left': -1, 'right': 1}).fillna(1)
        
        # side_multiplier: Home = -1, Away = 1
        # shooter is correctly identified as is_home or is_away.
        is_home_ser = (df['is_home'] == 1)
        side_mult = np.where(is_home_ser.values, -1, 1)
        
        attacking_side = def_side_sign.values * side_mult
        mask_flip = (attacking_side == -1)
        
        if not np.any(mask_flip):
            vprint("  [Standardization] Pass has 0 flips.")

        if np.any(mask_flip):
            vprint(f"  Flipping {np.sum(mask_flip)} events to Right-Attack orientation.")
            
            # Flip standard coordinates
            df.loc[mask_flip, 'x'] *= -1
            df.loc[mask_flip, 'y'] *= -1
            
            # CRITICAL: Flip adjusted coordinates in sync
            if 'x_adj' in df.columns:
                df.loc[mask_flip, 'x_adj'] *= -1
            if 'y_adj' in df.columns:
                df.loc[mask_flip, 'y_adj'] *= -1
            
            # Update metadata
            df.loc[mask_flip & is_home_ser, 'home_team_defending_side'] = 'left'
            df.loc[mask_flip & ~is_home_ser, 'home_team_defending_side'] = 'right'



    # 4. Arena Adjustments
    use_x, use_y = 'x', 'y'
    if apply_arena_adjustments:
        if 'x_adj' in df.columns and 'y_adj' in df.columns:
            use_x, use_y = 'x_adj', 'y_adj'
        elif 'home_team' in df.columns and 'season' in df.columns:
            vprint("  Calculating Arena Adjustments...")
            df['x_adj'] = df['x'].copy()
            df['y_adj'] = df['y'].copy()
            
            adj_map = arena_adjustments.load_adjustments()
            groups = df.groupby(['season', 'home_team'])
            for key, idxs in groups.groups.items():
                if not isinstance(key, tuple) or len(key) < 2:
                    continue
                season, team = key
                arena_name = arena_adjustments.resolve_arena(str(team))
                if season in adj_map and arena_name in adj_map[season]:
                     bias = adj_map[season][arena_name]
                     df.loc[idxs, 'x_adj'] -= bias.get('x_bias', 0.0)
                     df.loc[idxs, 'y_adj'] -= bias.get('y_bias', 0.0)
            use_x, use_y = 'x_adj', 'y_adj'

    # 5. Imputation
    if apply_imputation:
        vprint(f"  Imputing blocked shots using {use_x}, {use_y}...")
        df = impute.impute_blocked_shot_origins(
            df, 
            method='empirical_model', 
            x_col=use_x, 
            y_col=use_y, 
            is_standardized=True,
            alpha=impute_alpha
        )
        # Merge imputed into adj
        if 'imputed_x' in df.columns:
            blk_mask = (df['event'] == 'blocked-shot')
            if blk_mask.any():
                if 'x_adj' not in df.columns:
                    df['x_adj'] = df['x'].copy()
                    df['y_adj'] = df['y'].copy()
                df.loc[blk_mask, 'x_adj'] = df.loc[blk_mask, 'imputed_x']
                df.loc[blk_mask, 'y_adj'] = df.loc[blk_mask, 'imputed_y']

    # 5.5 Global Dithering
    if apply_dithering:
        if 'x_adj' not in df.columns:
            df['x_adj'] = df['x'].copy()
            df['y_adj'] = df['y'].copy()
        rng = np.random.default_rng(42)
        df['x_adj'] += rng.uniform(-0.5, 0.5, size=len(df))
        df['y_adj'] += rng.uniform(-0.5, 0.5, size=len(df))

    # 6. Final Swap & Features
    if 'x_adj' in df.columns:
        df['x'] = df['x_adj']
        df['y'] = df['y_adj']
    
    # Recalculate Distance/Angle
    # Forced float cast via string to bypass Categorical/ExtensionArray linter confusion
    c_x = pd.to_numeric(df['x'].astype(str), errors='coerce').fillna(0).astype(float).values
    c_y = pd.to_numeric(df['y'].astype(str), errors='coerce').fillna(0).astype(float).values
    
    # Standard Goal placement for standardized frame
    goal_x = 89.0
    
    # Use centralized rink logic to ensure dashboard alignment (90 deg = center)
    from . import rink
    new_dists = []
    new_angles = []
    for i in range(len(c_x)):
        d, a = rink.calculate_distance_and_angle(c_x[i], c_y[i], goal_x, 0.0)
        new_dists.append(d)
        new_angles.append(a)
    
    df['distance'] = pd.Series(new_dists, index=df.index)
    df['angle_deg'] = pd.Series(new_angles, index=df.index)

    # 6.5 REBOUND RECALCULATION
    # [AUDIT FIX] Apply stricter rebound definition (Save-only) to all data.
    # IMPORTANT: This logic MUST match puck/parse.py for ingestion consistency.
    # Definition: is_rebound=1 ONLY if last shot was 'shot-on-goal' within 5s.
    if 'team_id' in df.columns:
        df = df.sort_values(['game_id', 'period_number', 'total_time_elapsed_s'])
        df['is_rebound'] = 0
        df['rebound_source'] = 'none'
        
        # Use groupby team_id and game_id to find previous same-team shots
        groups = df.groupby(['game_id', 'team_id', 'period_number'])
        for _, group_idx in groups.groups.items():
            sub = df.loc[group_idx]
            if len(sub) < 2:
                continue
            
            prev_event = sub['event'].shift(1)
            prev_time = sub['total_time_elapsed_s'].shift(1)
            prev_angle = sub['angle_deg'].shift(1)
            time_diff = sub['total_time_elapsed_s'] - prev_time
            
            is_reb_mask = (time_diff <= 5.0) & (prev_event == 'shot-on-goal')
            
            df.loc[group_idx, 'is_rebound'] = is_reb_mask.astype(int)
            df.loc[group_idx, 'rebound_source'] = prev_event.fillna('none')
            df.loc[group_idx, 'rebound_time_diff'] = time_diff.fillna(0.0)
            
            ang_diff = (sub['angle_deg'] - prev_angle).abs()
            df.loc[group_idx, 'rebound_angle_change'] = ang_diff.fillna(0.0)

        # 6.6 SEQUENCE RECALCULATION
        # [AUDIT FIX] Recalculate sequence features using updated (standardized/imputed) coordinates.
        # This ensures that training features (previously from block location) match 
        # inference features (calculated from shooter location).
        
        # We already sorted by game/time above.
        # We need the last coordinate-bearing event for each game.
        df['dist_from_last_event'] = np.nan
        df['speed_from_last_event'] = np.nan
        df['angle_change_last_event'] = np.nan
        
        # Group by game and period to ensure sequence integrity
        for (gid, per), group_idx in df.groupby(['game_id', 'period_number']).groups.items():
            # We need to iterate because each event depends on the one before it
            # and we only want to track events that HAD coordinates.
            last_x, last_y = None, None
            last_t = None
            
            for idx in group_idx:
                curr_x = df.at[idx, 'x']
                curr_y = df.at[idx, 'y']
                curr_t = df.at[idx, 'total_time_elapsed_s']
                
                if last_x is not None and not np.isnan(curr_x):
                    dt = curr_t - last_t
                    dist = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
                    df.at[idx, 'dist_from_last_event'] = dist
                    if dt > 0.01:
                        df.at[idx, 'speed_from_last_event'] = dist / dt
                    
                    # Angle Change
                    _, last_ang = rink.calculate_distance_and_angle(last_x, last_y, goal_x, 0.0)
                    curr_ang = df.at[idx, 'angle_deg']
                    if not np.isnan(last_ang) and not np.isnan(curr_ang):
                        diff = abs(curr_ang - last_ang) % 360.0
                        if diff > 180.0: diff = 360.0 - diff
                        df.at[idx, 'angle_change_last_event'] = diff
                        
                # Update last event tracker if current event has coordinates
                if not np.isnan(curr_x):
                    last_x, last_y = curr_x, curr_y
                    last_t = curr_t

    # 7. Filtering
    if apply_filtering:
        shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
        df = df[df['event'].isin(shot_events)]
        if 'is_net_empty' in df.columns:
            df = df[df['is_net_empty'] != 1]

    # 8. Bio enrichment
    if apply_bio_enrichment:
        df = _enrich_bios_if_needed(df, verbose=verbose)

    # 9. Format
    df = _format_features(df, verbose=verbose)
    return df

def _enrich_bios_if_needed(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Enrich dataframe with player bio data (handedness and role) from the API."""
    from . import nhl_api
    
    # We want to enrich if either is missing or has NaNs
    needs_handedness = 'shoots_catches' not in df.columns or df['shoots_catches'].isna().any()
    needs_role = 'shooter_role' not in df.columns or df['shooter_role'].isna().any()
    
    if (needs_handedness or needs_role) and 'season' in df.columns and 'player_id' in df.columns:
        seasons = df['season'].unique()
        for s in seasons:
            if pd.isna(s): continue
            
            if verbose:
                print(f"  Fetching player bios for season {s}...")
            bios = nhl_api.get_season_player_bios(str(int(s)))
            if not bios:
                continue
            
            # Apply to rows matching this season
            mask = (df['season'] == s)
            # Fix: Convert float IDs (e.g. 8475181.0) to clean strings (e.g. '8475181')
            pids_raw = df.loc[mask, 'player_id']
            pids = pids_raw.dropna().astype(int).astype(str)
            
            if 'shoots_catches' not in df.columns:
                df['shoots_catches'] = np.nan
            if 'shooter_role' not in df.columns:
                df['shooter_role'] = np.nan
            
            # Map handedness
            handedness = pids.map(lambda x: bios.get(x, {}).get('shootsCatches'))
            df.loc[pids.index, 'shoots_catches'] = df.loc[pids.index, 'shoots_catches'].fillna(handedness)
            
            # Map role (Position Code)
            roles = pids.map(lambda x: bios.get(x, {}).get('positionCode'))
            # Standardize to F/D
            roles = roles.map(lambda x: 'D' if x == 'D' else ('F' if x in ['L', 'R', 'C'] else x))
            df.loc[pids.index, 'shooter_role'] = df.loc[pids.index, 'shooter_role'].fillna(roles)

    # Defaults
    if 'shoots_catches' not in df.columns:
        df['shoots_catches'] = 'Unknown'
    if 'shooter_role' not in df.columns:
         df['shooter_role'] = 'Unknown'
         
    return df



def _format_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Final formatting and default filling."""
    # Categoricals
    cat_cols = features.SHOT_TYPE + features.HANDEDNESS + features.PLAYER_ROLE + \
               ['last_event_type', 'game_state', 'relative_game_state', 'period_time_type']
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
        df[col] = df[col].fillna('Unknown').astype(str)
    
    # Numerics
    for col, default in NUMERIC_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        
    # Season (Special handling: force int)
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(20252026).astype(int)
    else:
        df['season'] = 20252026
        
    return df
