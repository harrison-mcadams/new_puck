import pandas as pd
import logging

logger = logging.getLogger(__name__)

def verify_df(df: pd.DataFrame, features: list, verify_blocked: bool = False):
    """
    Evaluates the dataframe prior to model fitting.
    Checks every feature for sensible values, missing data, and logs the results.
    """
    logger.info("=== verify_df: Checking Data Quality Before Model Fit ===")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Features count: {len(features)}")
    
    missing_features = []
    total_nans = 0
    total_cells = 0
    
    for feat in features:
        if feat not in df.columns:
            logger.error(f"  [ERROR] Feature '{feat}' is completely missing from the DataFrame!")
            missing_features.append(feat)
            continue
            
        series = df[feat]
        missing = series.isna().sum()
        total_nans += missing
        total_cells += len(df)
        pct_missing = (missing / len(df)) * 100
        
        dtype = series.dtype
        unique_vals = series.nunique(dropna=True)
        
        log_str = f"  Feature '{feat}' ({dtype}): {pct_missing:.2f}% missing | {unique_vals} unique vals"
        
        if pd.api.types.is_numeric_dtype(series):
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            log_str += f" | min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}"
            
            # Warn on suspicious things
            if unique_vals == 1 and not series.isna().all():
                log_str += " [WARNING: Zero variance!]"
        else:
            top_vals = series.value_counts(normalize=True).head(3).to_dict()
            top_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in top_vals.items()])
            log_str += f" | Top: {top_str}"
            
        if pct_missing > 50:
            log_str += " [WARNING: >50% missing!]"
            
        logger.info(log_str)
        
    # --- Blocked Shot Sanity Checks ---
    blocked_sane = True
    blocked_issues = []
    has_blocked_data = False
    
    if verify_blocked and 'event' in df.columns:
        logger.info("  --- Blocked Shot Verification ---")
        df_blocked = df[df['event'] == 'blocked-shot']
        df_shots = df[df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])]
        
        if len(df_blocked) == 0:
            logger.warning("  [BLOCK CHECK] No blocked shots found in DataFrame!")
        else:
            has_blocked_data = True
            # 1. Team Attribution / Orientation Check
            if 'x' in df_blocked.columns:
                neg_x_count = (df_blocked['x'] < 0).sum()
                neg_x_pct = (neg_x_count / len(df_blocked)) * 100
                logger.info(f"  [BLOCK CHECK] Team Assignment: {neg_x_pct:.1f}% blocked shots in defensive zone (x < 0)")
                if neg_x_pct > 10:
                    logger.warning(f"  [WARNING] High percentage ({neg_x_pct:.1f}%) of blocked shots in defensive zone. Possible team attribution flip!")
                    blocked_sane = False
                    blocked_issues.append("Attribution Flip")
            
            # 2. Imputation Check (Distance)
            if 'distance' in df_blocked.columns and 'distance' in df_shots.columns:
                mean_dist_bl = df_blocked['distance'].mean()
                mean_dist_sh = df_shots['distance'].mean()
                logger.info(f"  [BLOCK CHECK] Distance Mean: Blocked={mean_dist_bl:.1f}ft, Actual Shots={mean_dist_sh:.1f}ft")
                if mean_dist_bl < 25:
                    logger.warning(f"  [WARNING] Blocked shot mean distance ({mean_dist_bl:.1f}ft) is very low. Imputation may have failed!")
                    blocked_sane = False
                    blocked_issues.append("Imputation Failed")
            
            # 3. Spread/Distribution Check (Shot Type)
            if 'shot_type' in df_blocked.columns:
                bl_types = df_blocked['shot_type'].value_counts(normalize=True).head(3).to_dict()
                sh_types = df_shots['shot_type'].value_counts(normalize=True).head(3).to_dict()
                logger.info(f"  [BLOCK CHECK] Shot Type Spread (Blocked): {bl_types}")
                logger.info(f"  [BLOCK CHECK] Shot Type Spread (Shots):   {sh_types}")
            
            # 4. Angle Spread Check
            if 'angle_deg' in df_blocked.columns:
                mean_ang_bl = df_blocked['angle_deg'].mean()
                std_ang_bl = df_blocked['angle_deg'].std()
                logger.info(f"  [BLOCK CHECK] Angle Spread (Blocked): mean={mean_ang_bl:.1f}, std={std_ang_bl:.1f}")

    logger.info("=== SUMMARY ===")
    overall_nan_pct = (total_nans / total_cells * 100) if total_cells > 0 else 0
    logger.info(f"Total Missing Values in Features: {total_nans} ({overall_nan_pct:.2f}%)")
    
    if verify_blocked and 'event' in df.columns:
        if not has_blocked_data:
            logger.info("Blocked Shots Sane: N/A (None found)")
        elif blocked_sane:
            logger.info("Blocked Shots Sane: YES")
        else:
            logger.info(f"Blocked Shots Sane: NO ({', '.join(blocked_issues)})")

    if missing_features:
        logger.error(f"=== verify_df: FAILED. Missing features: {missing_features} ===")
    else:
        logger.info("=== verify_df: COMPLETE ===")
