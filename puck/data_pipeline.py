"""data_pipeline.py

Centralized pipeline for preprocessing PBP data for Training and Inference.
Refactored from logic previously in scripts/train_xgboost_model.py and puck/analyze.py.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, List, Tuple

from . import correction, impute, arena_adjustments, features

def preprocess_features(df_input: pd.DataFrame, 
                        is_training: bool = False, 
                        verbose: bool = False,
                        apply_arena_adjustments: bool = True,
                        apply_imputation: bool = True,
                        apply_dithering: bool = False,
                        apply_filtering: bool = False,
                        apply_bio_enrichment: bool = True,
                        impute_alpha: float = 0.0) -> pd.DataFrame:
    """
    Apply standard preprocessing steps to the dataframe:
    # 1. Blocked Shot Attribution
    # DIAGNOSIS (2025-01-14): The raw data for blocked shots ALREADY attributes the event to the SHOOTER (Offense).
    # The previous logic assumed it was attributed to the BLOCKER (Defense) and swapped it.
    # We disable this step to preserve the correct 'team_id' (Shooter).
    # if 'event' in df.columns and (df['event'] == 'blocked-shot').any():
    #     vprint("  Correcting Ref: Blocked Shots (Disabled, trusting raw attribution)...")
    #     # df = correction.fix_blocked_shot_attribution(df).
    2. Standardize Orientation (Force Attack Right based on swapped ownership)
    3. Global Dithering (Optional/Training)
    4. Arena Adjustments (Calculate or Use Existing)
    5. Impute Blocked Shot Origins
    6. Coordinate Swap & Feature Recalculation
    7. Event Filtering (Optional - Remove non-shots/extreme states)
    8. Bio Enrichment (Optional - Add shoots_catches, shooter_role)
    9. Feature Formatting (Fill NaNs, Enforce Types)
    
    Args:
        df_input: Raw PBP dataframe.
        is_training: If True, defaults apply_dithering to True (unless overridden).
        verbose: Print debug info.
        apply_arena_adjustments: Whether to apply arena bias corrections.
        apply_imputation: Whether to impute blocked shots.
        apply_dithering: Whether to add random noise (for smoothing/training).
        apply_filtering: Whether to filter out non-shots and extreme situations (empty net, 1v0).
        apply_bio_enrichment: Whether to enrich with player bios (handedness, role). Won't overwrite existing data.
    
    Returns:
        pd.DataFrame: Processed dataframe with 'x', 'y' updated and features (distance, angle) recalculated.
    """
    
    # Work on copy
    df = df_input.copy()
    
    if len(df) == 0:
        return df

    def vprint(*args):
        if verbose:
            print(*args)

    vprint(f"Preprocessing {len(df)} rows. Training={is_training}, Impute={apply_imputation}, Adjust={apply_arena_adjustments}")
    
    # Standardize Event Names to Lowercase immediately
    if 'event' in df.columns:
        df['event'] = df['event'].astype(str).str.lower()
        
        # Map to pipeline standard (handles spaces vs hyphens and 'shot' vs 'shot-on-goal')
        # This ensures 'Shot' -> 'shot' -> 'shot-on-goal' (Saved)
        evt_map = {
            'shot': 'shot-on-goal',
            'shots': 'shot-on-goal',
            'missed shot': 'missed-shot',
            'blocked shot': 'blocked-shot',
            'goal': 'goal'
        }
        # Only map values present in the map, preserve others (like existing hyphenated ones)
        df['event'] = df['event'].replace(evt_map)

    # 1. Blocked Shot Attribution
    # NOTE: The raw data for blocked shots attributes the event to the SHOOTER (Offense).
    # Previous versions of this pipeline attempted to swap this, assuming it was attributed to the blocker.
    # Verification (Jan 2025) confirmed that 'team_id' correctly points to the shooting team.
    # Therefore, no manual attribution swap is required here.


    # 2. Standardize Orientation (Attack Right)
    # Ensure all play is oriented towards the goal at x=89.
    # We must determine the True Attacking Side for each event.
    
    # Check if 'x' exists
    if 'x' not in df.columns:
        warnings.warn("Column 'x' missing in dataframe. skipping orientation standardization.")
    else:
        # Determine Coordinate Flip based on Attacking Side
        # Goal: Attacking Side should be RIGHT (Positive X).
        # Logic:
        # 1. Determine which side the Team is Attacking.
        # 2. If Team is Attacking Left (-X), Flip Everything.
        # 3. If Team is Attacking Right (+X), Do Nothing.
        
        # Prerequisites: 'team_id', 'home_id', 'home_team_defending_side'
        # Note: 'team_id' tracks the "Performer" (Shooter for shots, Blocker for blocks due to prev logic)
        # We assume 'team_id' is the Attacking Team for the event context.
        
        can_determine_side = ('team_id' in df.columns and 
                              'home_id' in df.columns and 
                              'home_team_defending_side' in df.columns)
                              
        if can_determine_side:
            # Vectorized Logic
            
            # Map side string to sign (-1 = Left, +1 = Right)
            # 'left' -> -1, 'right' -> 1
            # Careful with case/whitespace
            # --- 2. Standardize Orientation (Attack Right) ---
            # Ensure all play is oriented towards the goal at x=89.
            
            # Robust extraction of side map
            # normalize to 'left' or 'right'
            side_str = df['home_team_defending_side'].astype(str).str.lower().str.strip()
            def_side_map = side_str.map({'left': -1, 'right': 1})
            
            # Robust Is_Home check (Force String Comparison)
            # handle NaNs gracefully
            tid_str = df['team_id'].fillna(-1).astype(str).str.split('.').str[0] # Handle float strings e.g. "6.0"
            hid_str = df['home_id'].fillna(-2).astype(str).str.split('.').str[0]
            is_home_series = (tid_str == hid_str)
            
            # Side Multiplier: Home = -1, Away = 1
            # Logic: If Home (-1 side) * Home (-1 mult) = +1 Attack
            side_multiplier = np.where(is_home_series, -1, 1)
            
            # Attacking Side = Def Side * Multiplier
            attacking_side = def_side_map * side_multiplier
            
            # Filter rows where Attack is Left (-1)
            mask_flip = (attacking_side == -1)
            
            if mask_flip.any():
                vprint(f"  Flipping {mask_flip.sum()} events (attacking left).")
                df.loc[mask_flip, 'x'] *= -1
                df.loc[mask_flip, 'y'] *= -1
            
            # Update Copies if present (though usually created later)
            if 'x_adj' in df.columns:
                 df['x_adj'] = df['x']
            if 'y_adj' in df.columns:
                 df['y_adj'] = df['y']
                
            # CRITICAL: Synchronize Metadata
            if 'home_team_defending_side' in df.columns:
                 df.loc[mask_flip & is_home_series, 'home_team_defending_side'] = 'left'
                 df.loc[mask_flip & ~is_home_series, 'home_team_defending_side'] = 'right'
                    
        else:
             # FALLBACK: Old simplistic logic
             # Assume all redundant negative X events are offensive zone
             vprint("  WARNING: Cannot determine true attacking side (missing cols). Using blind flip.")
             mask_neg = df['x'] < 0
             if mask_neg.any():
                vprint(f"  Flipping {mask_neg.sum()} events to positive orientation.")
                df.loc[mask_neg, 'x'] *= -1
                df.loc[mask_neg, 'y'] *= -1
                if 'x_adj' in df.columns:
                    df.loc[mask_neg, 'x_adj'] *= -1
                if 'y_adj' in df.columns:
                    df.loc[mask_neg, 'y_adj'] *= -1
                
                # Update metadata for fallback as well
                if 'home_team_defending_side' in df.columns:
                    df.loc[mask_neg, 'home_team_defending_side'] = 'left'



    
    # 4. Arena Adjustments
    use_x, use_y = 'x', 'y'
    
    if apply_arena_adjustments:
        # OPTION A: Use Pre-Calculated Adjustments (Preferred)
        if 'x_adj' in df.columns and 'y_adj' in df.columns:
            vprint("  Found existing 'x_adj' columns. Using them.")
            use_x, use_y = 'x_adj', 'y_adj'
            
            # Valid negative x_adj (Defensive Zone) should NOT be flipped if we trust inputs
            # Re-standardize logic removed to prevent mirroring defensive zone blocks
            # mask_neg_adj = df['x_adj'] < 0
            # if mask_neg_adj.any():
            #    vprint(f"    Re-standardizing {mask_neg_adj.sum()} x_adj values.")
            #    df.loc[mask_neg_adj, 'x_adj'] *= -1
            #    df.loc[mask_neg_adj, 'y_adj'] *= -1
                
        # OPTION B: Calculate from Team Name
        elif 'home_team' in df.columns or 'home_abb' in df.columns:
            vprint("  Calculating Arena Adjustments from team info...")
            
            # Create working columns
            df['x_adj'] = df['x'].copy()
            df['y_adj'] = df['y'].copy()
            
            # Vectorized approach using arena_adjustments module logic might be slow if we loop rows.
            # But train_xgboost_model.py had a fast GroupBy approach. Let's reuse that.
            
            adj_map = arena_adjustments.load_adjustments()
            
            # Determine suitable columns
            c_team = 'home_team' if 'home_team' in df.columns else 'home_abb'
            c_season = 'season' if 'season' in df.columns else None
            # If no season col, maybe cannot adjust? Or assume current?
            # Default to no adjustment if no season
            
            if c_season:
                 # Standardize team names
                temp_team = df[c_team].map(arena_adjustments.resolve_arena)
                temp_season = df[c_season].astype(str)
                
                # Apply map
                # Iterate over unique Season/Arena combos
                groups = df.groupby([temp_season, temp_team])
                count_adj = 0
                
                for (season_val, arena_val), idxs in groups.groups.items():
                    if season_val in adj_map and arena_val in adj_map[season_val]:
                         dx = adj_map[season_val][arena_val].get('x_bias', 0.0)
                         dy = adj_map[season_val][arena_val].get('y_bias', 0.0)
                         if dx != 0 or dy != 0:
                             df.loc[idxs, 'x_adj'] = df.loc[idxs, 'x'] - dx
                             df.loc[idxs, 'y_adj'] = df.loc[idxs, 'y'] - dy
                             count_adj += 1
                
                vprint(f"    Applied adjustments to {count_adj} groups.")
                
                # Re-standardize check removed
                # mask_neg_adj = df['x_adj'] < 0
                # if mask_neg_adj.any():
                #    df.loc[mask_neg_adj, 'x_adj'] *= -1
                #    df.loc[mask_neg_adj, 'y_adj'] *= -1
                    
                use_x, use_y = 'x_adj', 'y_adj'
            else:
                 vprint("    WARNING: 'season' column missing. Cannot apply Arena Adjustments.")
        else:
            vprint("    WARNING: No 'home_team' or 'x_adj'. Skipping Arena Adjustments.")

    # 5. Imputation
    if apply_imputation:
        vprint(f"  Imputing blocked shots using {use_x}, {use_y}...")
        # Note: impute.py applies its own internal dithering to blocked shots to smooth
        # the discrete NHL API coordinates for better model lookup. We accept this.
        
        
        df = impute.impute_blocked_shot_origins(
            df, 
            method='empirical_model', 
            x_col=use_x, 
            y_col=use_y, 
            is_standardized=True,
            alpha=impute_alpha
        )
    
    # Merge Imputed into Adjusted
    # If imputation ran, 'imputed_x' contains valid coords for blocked shots.
    # We want x_adj to reflect this for those rows.
    if 'imputed_x' in df.columns:
         # Where imputed_x is not null (meaning it was imputed), update x_adj
         # Note: imputed_x might be full copy of x_col? 
         # impute.py: df_out['imputed_x'] = df_out[x_col]
         # So yes, it's safe to just use imputed_x as the new x_adj?
         # Or only for blocked shots?
         # User said: "blocked shots now also have an appropriate x_adj"
         # Let's target blocked shots specifically to be safe/clear.
         mask_blk = (df['event'] == 'blocked-shot')
         if mask_blk.any():
              vprint("  Merging imputed coordinates into x_adj for blocked shots...")
              df.loc[mask_blk, 'x_adj'] = df.loc[mask_blk, 'imputed_x']
              df.loc[mask_blk, 'y_adj'] = df.loc[mask_blk, 'imputed_y']
              
              # PBP block locations ('block_x') are preserved by impute function automatically

    # 5.5 Global Dithering (Moved per user request)
    # Apply to x_adj/y_adj BEFORE swap, so it affects features and standardizes precision.
    if apply_dithering:
        vprint("  Applying Global Dithering (+/- 0.5ft) to Adjusted Coordinates...")
        
        # Ensure x_adj/y_adj exist (if not created by adjustments/imputation)
        if 'x_adj' not in df.columns:
            df['x_adj'] = df['x']
        if 'y_adj' not in df.columns:
            df['y_adj'] = df['y']
            
        rng = np.random.default_rng(42)
        noise_x = rng.uniform(-0.5, 0.5, size=len(df))
        noise_y = rng.uniform(-0.5, 0.5, size=len(df))
        
        df['x_adj'] = df['x_adj'] + noise_x
        df['y_adj'] = df['y_adj'] + noise_y

    # 6. Coordinate Swap & Feature Recalculation
    # "takes x_adj and y_adj, swaps them over into the x and y columns"
    
    # Ensure x_adj/y_adj exist (if skipped above)
    if 'x_adj' not in df.columns:
        df['x_adj'] = df['x']
        df['y_adj'] = df['y']
        
    vprint("  Swapping Adjusted Coordinates into Main Columns (x,y)...")
    df['x'] = df['x_adj']
    df['y'] = df['y_adj']
    
    # CRITICAL: Must use 'x_adj' / 'y_adj' if available, as these contain:
    # A) Arena Adjustments
    # B) Imputed Origins (for blocked shots)
    # Using raw 'x' would calculate metrics to the Block Location, causing leakage.
    
    calc_x = pd.to_numeric(df['x_adj'] if 'x_adj' in df.columns else df['x'], errors='coerce')
    calc_y = pd.to_numeric(df['y_adj'] if 'y_adj' in df.columns else df['y'], errors='coerce')
    
    # x_adj is Standardized to Right Attack (Net at 89)
    # y_adj is Standardized (-42.5 to 42.5)
    
    goal_x = 89.0
    df['distance'] = np.sqrt((calc_x - goal_x)**2 + calc_y**2)
    
    # Angle
    dx = calc_x - goal_x
    dy = calc_y
    # Fixed CCW calc from vectors
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    df['angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0

    # 7. Event Filtering (Optional)
    if apply_filtering:
        vprint("  Applying Event Filtering (Step 7)...")
        initial_len = len(df)
        
        # A. Keep only Shot Events
        shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
        if 'event' in df.columns:
            df = df[df['event'].isin(shot_events)]
            
        # B. Remove Empty Net
        if 'is_net_empty' in df.columns:
            # removing rows where is_net_empty == 1
            df = df[df['is_net_empty'] != 1]


        # C. Remove Extreme Game States (1v0, 0v1)
        if 'game_state' in df.columns:
             df = df[~df['game_state'].isin(['1v0', '0v1'])]
             
        vprint(f"    Filtered {initial_len - len(df)} rows. Final count: {len(df)}")

    # 8. Bio Enrichment (Optional)
    if apply_bio_enrichment:
        df = _enrich_bios_if_needed(df, verbose=verbose)

    # 9. Feature Formatting (Fill NaNs, Enforce Types)
    df = _format_features(df, verbose=verbose)

    return df

def _enrich_bios_if_needed(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add player handedness (shoots_catches) and role (shooter_role) if missing or empty.
    
    Only enriches if:
    - Column doesn't exist
    - Column is all NaN
    - Column is all 'Unknown'
    """
    if df.empty:
        return df
    
    def vprint(*args):
        if verbose:
            print(*args)
    
    # Check if enrichment is needed
    needs_shoots_catches = (
        'shoots_catches' not in df.columns or
        df['shoots_catches'].isna().all() or
        (df['shoots_catches'].astype(str).str.upper() == 'UNKNOWN').all()
    )
    
    needs_shooter_role = (
        'shooter_role' not in df.columns or
        df['shooter_role'].isna().all() or
        (df['shooter_role'].astype(str).str.upper() == 'UNKNOWN').all()
    )
    
    if not needs_shoots_catches and not needs_shooter_role:
        vprint("  Bio columns already populated. Skipping enrichment.")
        return df
    
    # Need player_id and game_id to enrich
    if 'player_id' not in df.columns or 'game_id' not in df.columns:
        vprint("  Missing player_id or game_id. Cannot enrich bios. Using NaN for marginalization.")
        if needs_shoots_catches:
            df['shoots_catches'] = np.nan
        if needs_shooter_role:
            df['shooter_role'] = np.nan
        return df
    
    vprint("  Enriching Bio Data (shoots_catches, shooter_role)...")
    
    try:
        from . import nhl_api
        
        # Derive season from game_id
        df['_temp_season_start'] = df['game_id'].astype(str).str[:4]
        mask_valid = df['_temp_season_start'].str.isdigit()
        unique_starts = df.loc[mask_valid, '_temp_season_start'].astype(int).unique()
        
        master_map = {}
        for start_year in unique_starts:
            if start_year < 1900 or start_year > 2100:
                continue
            season_str = f"{start_year}{start_year + 1}"
            try:
                bios = nhl_api.get_season_player_bios(season_str)
                master_map.update(bios)
            except Exception as e:
                vprint(f"    Warning: Failed to fetch bios for {season_str}: {e}")
        
        # Drop temp column
        df.drop(columns=['_temp_season_start'], inplace=True, errors='ignore')
        
        if not master_map:
            vprint("    No bios loaded. Using NaN for marginalization.")
            if needs_shoots_catches:
                df['shoots_catches'] = np.nan
            if needs_shooter_role:
                df['shooter_role'] = np.nan
            return df
        
        # Helper to get value from bio map
        def get_bio_val(pid_val, field, default=None):
            if pd.isna(pid_val):
                return default
            try:
                clean_id = str(int(float(pid_val)))
            except:
                clean_id = str(pid_val)
            entry = master_map.get(clean_id)
            if not entry:
                return default
            return entry.get(field, default)
        
        # Enrich only if needed
        if needs_shoots_catches:
            df['shoots_catches'] = df['player_id'].apply(lambda x: get_bio_val(x, 'shootsCatches', np.nan))
            vprint(f"    Enriched 'shoots_catches' for {len(df)} rows.")
        
        if needs_shooter_role:
            def map_role(pos_code):
                if not pos_code:
                    return np.nan  # Let marginalization handle unknowns
                return 'D' if pos_code == 'D' else 'F'
            
            df['shooter_role'] = df['player_id'].apply(lambda x: map_role(get_bio_val(x, 'positionCode')))
            vprint(f"    Enriched 'shooter_role' for {len(df)} rows.")
            
    except Exception as e:
        warnings.warn(f"Bio enrichment failed: {e}. Using NaN for marginalization.")
        if needs_shoots_catches and 'shoots_catches' not in df.columns:
            df['shoots_catches'] = np.nan
        if needs_shooter_role and 'shooter_role' not in df.columns:
            df['shooter_role'] = np.nan
    
    
    return df

def _format_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Ensure all expected features exist and are correctly formatted.
    Fills missing values with defaults ('Unknown' or 0) and casts types.
    """
    if df.empty:
        return df
        
    if verbose:
        print("  Formatting Features (Step 8)...")
    
    # 1. Categoricals
    # Use lists from features.py where available, plus common metadata
    categorical_cols = features.SHOT_TYPE + features.HANDEDNESS + features.PLAYER_ROLE + \
                       ['last_event_type', 'game_state', 'period_time_type', 'home_team_defending_side', 
                        'player_name', 'team_abbrev', 'home_abb', 'away_abb']
                        
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            # Object/String columns can have None/NaN
            df[col] = df[col].fillna('Unknown')
        
        # Cast to string to be safe for categorical encoding later
        df[col] = df[col].astype(str)

    # 2. Numerics
    # Fill missing numeric features with appropriate defaults (usually 0)
    numeric_defaults = {
        'is_rebound': 0,
        'rebound_angle_change': 0.0,
        'rebound_time_diff': 0.0,
        'is_rush': 0,
        'last_event_time_diff': 0.0,
        'score_diff': 0,
        # Ensure coordinates/angles are at least present (though should be calc'd)
        'distance': -1.0, 
        'angle_deg': 0.0,
        # Time
        'time_elapsed_in_period_s': 0.0,
        'total_time_elapsed_s': 0.0,
        'dist_from_last_event': 0.0,
        'speed_from_last_event': 0.0
    }
    
    for col, default_val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)
            
    return df
