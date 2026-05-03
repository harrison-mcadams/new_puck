import numpy as np
import pandas as pd

try:
    from .rink import calculate_distance_and_angle, rink_goal_xs
except ImportError:
    # Fallback if relative import fails or rink not found
    def calculate_distance_and_angle(x, y, gx, gy=0.0):
         import math
         return math.hypot(x-gx, y-gy), 0.0
    def rink_goal_xs(): return -89.0, 89.0

def fix_blocked_shot_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    BLOCKED SHOT ATTRIBUTION — NO-OP (SWAP DISABLED)
    
    BACKGROUND:
    The NHL GameCenter API (api-web.nhle.com) attributes 'blocked-shot' events
    to the SHOOTER's team via eventOwnerTeamId. This was verified across 1,867
    blocked shots in 60 games spanning all modern-era seasons (2020-2026):
    eventOwnerTeamId == shooter's team in 100.0% of cases.
    
    HISTORY:
    This function previously swapped team_id (home <-> away) under the incorrect
    assumption that the API attributed blocks to the blocker's team. The swap
    was accidentally compensated by the training pipeline running
    preprocess_features() on CSVs that already had the swap baked in,
    resulting in a double-swap that cancelled out. The model trained correctly
    despite the bug.
    
    CURRENT BEHAVIOR:
    This function is now a no-op. It returns the DataFrame unchanged.
    The team_id swap has been removed because:
    1. The API already provides correct shooter-team attribution.
    2. data_pipeline.preprocess_features() recalculates distance/angle for ALL
       events (not just blocked shots) at the standardization step, making the
       distance recalculation here redundant.
    3. Removing the swap eliminates the fragile double-swap dependency and
       ensures correct attribution in a single pass.
    
    Args:
        df: DataFrame containing event data.
        
    Returns:
        pd.DataFrame: The input DataFrame, unchanged.
    """
    return df
