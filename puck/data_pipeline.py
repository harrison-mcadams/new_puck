"""data_pipeline.py

Centralized pipeline for preprocessing PBP data for Training and Inference.
Refactored from logic previously in scripts/train_xgboost_model.py and puck/analyze.py.

ARCHITECTURE NOTE - COORDINATE FLOW:
1. Raw API Data (PBP) arrives with 'blocked-shots' attributed to the DEFENSE.
2. correction.fix_blocked_shot_attribution():
   - Swaps 'team_id' and 'event_owner_team_id' to the ATTACKER.
   - This ensures we are predicting "Will this player score?" NOT "Will this player block?".
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
    Historically, the NHL API attributes a 'blocked-shot' event to the team of the player 
    WHO BLOCKED the shot (the defense). This causes significant issues for xG models:
    1. Team Attribution: The shot is credited to the wrong team's statistics.
    2. Orientation: Coordinates appear on the wrong side of the ice (defensive zone).
    3. Shot Type: The API often omits the shot type (Wrist, Slap, etc.) for blocks.
    
    WE FIX THIS HERE BY:
    1. Swapping 'team_id' to the attacking team (the shooter).
    2. Standardizing coordinates to the attacker's 'scoring' orientation.
    3. Recovering 'shot_type' via fuzzy-matching with NHL HTML Play-by-Play reports.
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
        # 1. Step: Fix blocked shot attribution BEFORE standardization to be consistent
        if apply_attribution_fix and (df['event'].astype(str).str.lower() == 'blocked-shot').any():
            vprint("  Correcting Attribution: Blocked Shots (Blocker -> Shooter)...")
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

    # Derive is_home early (Always derive to reflect corrections like blocked swaps)
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
    if apply_html_enrichment and game_id:
        vprint(f"  Enriching shots with HTML PBP for game {game_id}...")
        df = html_enrichment.enrich_blocks_with_html(df, game_id)

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
        # NOTE: correction.py ALREADY swaps team_id to the SHOOTING team for blocked shots!
        # So is_home correctly identifies if the SHOOTING team is home.
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
    """Mock/Simplified Bio Enrichment if missing handedness."""
    # Real implementation would call nhl_api.get_season_player_bios
    # For now, we assume col exists or fill with Unknown
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
