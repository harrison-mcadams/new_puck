"""mixed_effects.py

GAME-STATE AWARE MIXED EFFECTS MODEL (JOINT OFF/DEF INTERCEPTS)
================================================================
This module implements a mixed effects model that fits random intercepts for
BOTH Offense (Team) and Defense (Opponent) simultaneously.

It handles Game State splitting (5v5, 5v4, 4v5) by training separate
sub-models for each state.

Architecture:
-------------
1. Base Model: NestedGLM (Fixed Effect) -> Provides P_base / Base Margin.
2. Mixed Effect: L-BFGS-B Logistic Solver
   - Goal: Fit `logit(p) = base_margin + Off_Intercept + Def_Intercept`
   - Off_Intercept = coef[off_team_idx]
   - Def_Intercept = coef[def_team_idx + n_teams]

Usage:
------
    mixed = GameMixedEffectsXG(base_model_path="...")
    mixed.fit(df)
    mixed.predict_proba(df)
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Dict, Optional, Union
from pathlib import Path
import scipy.sparse as sp

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 


# ---------------------------------------------------------------------------
# Legacy Aliases — strictly for unpickling old .joblib files.
# Do NOT use for new code.
# ---------------------------------------------------------------------------
class ParallelMixedEffectsModel:
    """Legacy stub for unpickling old models."""
    pass

class ComponentMixedEffectsModel:
    """Legacy stub for unpickling old models."""
    pass


def _derive_off_def_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive off_team_name and def_team_name columns using a single,
    consistent strategy.  Mutates and returns *df*.

    Priority:
      1. Numeric team_id == home_id comparison  (most reliable)
      2. String team_abbrev == home_abb fallback (if IDs unavailable)
    """
    if 'off_team_name' in df.columns and 'def_team_name' in df.columns:
        return df  # Already present

    has_ids = all(c in df.columns for c in ('team_id', 'home_id', 'home_abb', 'away_abb'))

    if has_ids:
        # Reliable numeric comparison
        is_home_shot = pd.to_numeric(df['team_id'], errors='coerce') == pd.to_numeric(df['home_id'], errors='coerce')
        df['off_team_name'] = np.where(is_home_shot, df['home_abb'], df['away_abb'])
        df['def_team_name'] = np.where(is_home_shot, df['away_abb'], df['home_abb'])
    else:
        # String fallback
        if 'team_abbrev' in df.columns:
            df['off_team_name'] = df['team_abbrev']
        elif 'team_id' in df.columns:
            df['off_team_name'] = df['team_id'].astype(str)
        else:
            df['off_team_name'] = 'Unknown'

        if 'home_abb' in df.columns and 'away_abb' in df.columns:
            off_vec = df['off_team_name'].astype(str)
            home_vec = df['home_abb'].astype(str)
            is_home = (off_vec == home_vec)
            df['def_team_name'] = np.where(is_home, df['away_abb'], df['home_abb'])
        else:
            df['def_team_name'] = 'Unknown'

    return df


class StateMixedEffectsModel(BaseEstimator):
    """
    Fits a joint mixed-effects model for a single game state (e.g., 5v5).
    Fits Random Intercepts for BOTH Offense and Defense simultaneously 
    to properly apportion credit/blame from the base marginal residual.
    """
    def __init__(self, 
                 l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.coef_ = None # Store coefficients from Scipy solver (size: 2 * N_teams)
        self.teams_ = None
        self.team_idx_ = None # Map[team_name -> int]
        self.n_teams_ = 0
        self.converged_ = False
        self.final_loss_ = None
        
    def fit(self, df: pd.DataFrame, y: pd.Series, base_margin: np.ndarray, 
            off_col: str, def_col: str):
        """
        Fit the joint mixed effects model.
        
        Args:
            df: Feature dataframe (must contain off_col and def_col)
            y: Target (0/1)
            base_margin: Log-odds from base model
            off_col: Column name for shooting team
            def_col: Column name for defending team
        """
        # 1. Identify all unique teams across both columns to ensure alignment
        off_teams = set(df[off_col].dropna().unique())
        def_teams = set(df[def_col].dropna().unique())
        all_teams = off_teams.union(def_teams)
        
        # Filter valid strings
        teams = [t for t in all_teams if isinstance(t, str) and len(t) > 0]
        teams.sort()
        
        self.teams_ = teams
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        self.n_teams_ = len(self.teams_)
        
        logger.info(f"State Model (Joint Intercepts): {len(df)} samples, {self.n_teams_} teams.")
        
        # 2. Map teams to indices
        off_indices = df[off_col].map(self.team_idx_)
        def_indices = df[def_col].map(self.team_idx_)
        
        # Filter invalid rows where team is missing
        valid_mask = (~off_indices.isna()) & (~def_indices.isna())
        if not valid_mask.all():
            n_dropped = int(np.sum(~valid_mask))
            logger.warning(f"Dropping {n_dropped} rows with missing team info.")
            off_indices = off_indices[valid_mask]
            def_indices = def_indices[valid_mask]
            y = y[valid_mask]
            base_margin = base_margin[valid_mask.values] if isinstance(valid_mask, pd.Series) else base_margin[valid_mask]
            
        n_samples = len(off_indices)
        off_idx_vals = off_indices.values.astype(int)
        def_idx_vals = def_indices.values.astype(int)
        
        # 3. Construct Unified Sparse Design Matrix
        # Size: (N_samples, 2 * N_teams)
        # Each row has exactly TWO 1.0s:
        # Col A = off_team_idx (Offense Intercept)
        # Col B = def_team_idx + n_teams (Defense Intercept)
        
        # Row indices (each row gets two entries)
        row_indices = np.repeat(np.arange(n_samples), 2)
        
        # Column indices
        col_indices = np.empty(2 * n_samples, dtype=int)
        col_indices[0::2] = off_idx_vals # Offense features (0 to N-1)
        col_indices[1::2] = def_idx_vals + self.n_teams_ # Defense features (N to 2N-1)
        
        # Data values (all 1.0 for intercepts)
        data = np.ones(2 * n_samples, dtype=np.float32)
        
        input_dim = 2 * self.n_teams_
        X_sparse = sp.coo_matrix((data, (row_indices, col_indices)), 
                                 shape=(n_samples, input_dim)).tocsr()
                                 
        # 4. Fit using logistic solver
        y_float = y.values.astype(np.float64) if hasattr(y, 'values') else np.asarray(y, dtype=np.float64)
        if hasattr(base_margin, 'values'):
             base_margin = base_margin.values
        base_margin = base_margin.reshape(-1).astype(np.float64)
        
        from puck.logistic_solver import fit_logistic_offset
        
        logger.info(f"Solving Joint State Model (L-BFGS-B)...")
        self.coef_, self.converged_, self.final_loss_ = fit_logistic_offset(
            X_sparse, y_float, base_margin, 
            l2_reg=self.l2_reg, verbose=False
        )
        
        if not self.converged_:
            logger.warning("Optimizer did NOT converge. Coefficients may be unreliable.")
        
        # 5. Sanity checks
        max_abs = np.max(np.abs(self.coef_))
        mean_abs = np.mean(np.abs(self.coef_))
        logger.info(f"Joint State Model Fit Complete. "
                     f"Converged={self.converged_}, Loss={self.final_loss_:.4f}, "
                     f"MaxAbsCoef={max_abs:.4f}, MeanAbsCoef={mean_abs:.4f}")
        
        if max_abs > 2.0:
            logger.warning(f"Large coefficient detected (max |coef| = {max_abs:.4f}). "
                           f"Consider increasing L2 regularization.")
        if mean_abs < 1e-6:
            logger.warning(f"All coefficients near zero (mean |coef| = {mean_abs:.6f}). "
                           f"Base model may already be well-calibrated — mixed effects add nothing.")
        
        return self

    def predict_margin(self, df: pd.DataFrame, off_col: str, def_col: str) -> np.ndarray:
        """
        Predict the joint margin adjustment (Offense + Defense).
        Returns: Adjustment vector (N_samples,)
        """
        if self.coef_ is None:
            return np.zeros(len(df))
            
        n_samples = len(df)
        
        # Map indices
        off_idx_raw = df[off_col].map(self.team_idx_)
        def_idx_raw = df[def_col].map(self.team_idx_)
        
        off_adj = np.zeros(n_samples)
        def_adj = np.zeros(n_samples)
        
        off_valid = ~off_idx_raw.isna()
        if off_valid.any():
            off_vals = off_idx_raw.values[off_valid].astype(int)
            off_adj[off_valid] = self.coef_[off_vals]
            
        def_valid = ~def_idx_raw.isna()
        if def_valid.any():
            def_vals = def_idx_raw.values[def_valid].astype(int)
            def_adj[def_valid] = self.coef_[def_vals + self.n_teams_]
            
        return off_adj + def_adj

    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract the learned intercepts into a readable DataFrame.
        """
        if self.coef_ is None:
            return pd.DataFrame()
            
        records = []
        for t_i, team in enumerate(self.teams_):
            # Offense Profile
            records.append({
                'team': team,
                'role': 'Offense',
                'feature': 'intercept',
                'coef': self.coef_[t_i]
            })
            # Defense Profile
            records.append({
                'team': team,
                'role': 'Defense',
                'feature': 'intercept',
                'coef': self.coef_[t_i + self.n_teams_]
            })
            
        return pd.DataFrame(records)


class GameMixedEffectsXG(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 base_model_path: str = None, 
                 base_model = None,
                 feature_set: Union[List[str], str] = None,
                 use_tensor_splines: bool = False,
                 updater: str = 'shotgun',
                 component_model_type: str = 'intercept',
                 l2_reg: float = 1.0
                 ):
        
        self.base_model_path = base_model_path
        self.base_model_ = base_model
        self.feature_set = feature_set
        self.use_tensor_splines = use_tensor_splines
        self.updater = updater 
        self.component_model_type = component_model_type
        self.l2_reg = l2_reg
        
        # Sub-models per game state (Joint Off/Def)
        self.state_models_: Dict[str, StateMixedEffectsModel] = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # 1. Load Base Model
        if self.base_model_ is None:
            if self.base_model_path is None:
                # Default path
                p = Path("analysis/xgs/xg_model_nested_tensor.joblib")
                if p.exists():
                     self.base_model_path = str(p)
                else:
                    raise FileNotFoundError("Base model not found/specified.")
            
            logger.info(f"Loading Base Model: {self.base_model_path}")
            self.base_model_ = joblib.load(self.base_model_path)
            
        # 2. Predict Base Margins
        logger.info("Predicting Base Margins...")
        
        # Ensure required columns for base model exist
        required_cols = ['shoots_catches', 'shooter_role', 'is_rebound', 'is_rush']
        for c in required_cols:
            if c not in df.columns:
                logger.warning(f"Column '{c}' missing for base model. Filling with default.")
                if c == 'shoots_catches':
                    df[c] = 'L' 
                elif c == 'shooter_role':
                    df[c] = 'Center' 
                else:
                    df[c] = 0

        # Derive off/def team names (unified logic)
        df = _derive_off_def_names(df)

        base_probs = self.base_model_.predict_proba(df)[:, 1]
        
        logger.info(f"Base Model Stats (Fit): Mean={base_probs.mean():.4f}, "
                     f"Min={base_probs.min():.4f}, Max={base_probs.max():.4f}")
        
        eps = 1e-6
        base_probs = np.clip(base_probs, eps, 1-eps)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        # Convert to Series for index-safe subsetting later
        base_margins = pd.Series(base_margins, index=df.index)
        
        # 3. Train per Game State
        states = df['game_state'].value_counts()
        target_states = ['5v5', '5v4', '4v5']
        valid_states = [s for s in target_states if s in states.index and states[s] > 100]
        logger.info(f"Training models for states: {valid_states}")
        
        # Warn about unmodeled states
        unmodeled = set(df['game_state'].unique()) - set(valid_states)
        if unmodeled:
            n_unmodeled = int(df['game_state'].isin(unmodeled).sum())
            logger.warning(f"Game states {unmodeled} have no mixed-effects model. "
                           f"{n_unmodeled} shots will use base xG only.")
        
        for state in valid_states:
            logger.info(f"--- Fitting State: {state} ---")
            mask = df['game_state'] == state
            df_sub = df[mask]
            if len(df_sub) == 0:
                continue
            
            # Target
            y_sub = y[mask] if y is not None else (df_sub['event'] == 'goal').astype(int)
            margin_sub = base_margins[mask]
            
            # Joint Model
            logger.info(f"Fitting Joint Offense/Defense ({state})...")
            state_model = StateMixedEffectsModel(
                l2_reg=self.l2_reg
            )
            state_model.fit(df_sub, y_sub, margin_sub.values, 
                           off_col='off_team_name', def_col='def_team_name')
            self.state_models_[state] = state_model
            
        return self

    def predict_proba(self, X: pd.DataFrame):
        df = X.copy()
        
        # 1. Base
        base_probs = self.base_model_.predict_proba(df)[:, 1]
        
        logger.info(f"Base Model Stats: Mean={base_probs.mean():.4f}, "
                     f"Min={base_probs.min():.4f}, Max={base_probs.max():.4f}")
        
        eps = 1e-6
        base_probs = np.clip(base_probs, eps, 1-eps)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        final_margins = base_margins.copy()
        
        # Derive off/def team names (unified logic)
        df = _derive_off_def_names(df)

        # 2. Add Adjustments per State
        state_models = getattr(self, 'state_models_', {}) or {}
        
        for state, model in state_models.items():
            mask = df['game_state'] == state
            if not mask.any():
                continue
            
            adj = model.predict_margin(df[mask], off_col='off_team_name', def_col='def_team_name')
            final_margins[mask.values] += adj
            
        # 3. Sigmoid
        final_probs = 1.0 / (1.0 + np.exp(-final_margins))
        return np.column_stack((1 - final_probs, final_probs))
        
    def get_all_coefficients(self) -> pd.DataFrame:
        """
        Aggregate coefficients from all sub-models into a single DataFrame.
        """
        dfs = []
        for state, model in self.state_models_.items():
            df_curr = model.get_coefficients()
            if not df_curr.empty:
                df_curr['game_state'] = state
                dfs.append(df_curr)

        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def save_summary(self, output_dir: str, teams_filter: List[str] = None):
        """
        Save model coefficients and generate summary plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Get Data
        df_coefs = self.get_all_coefficients()
        if df_coefs.empty:
            logger.warning("No coefficients to save.")
            return
            
        # 2. Save CSV
        csv_path = output_dir / "mixed_effects_coefficients.csv"
        df_coefs.to_csv(csv_path, index=False)
        logger.info(f"Saved coefficients to {csv_path}")
        
        # 3. Generate Plots
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Could not import matplotlib/seaborn. Skipping plots.")
            return
        
        sns.set_theme(style="whitegrid")
        
        mask_intercept = df_coefs['feature'] == 'intercept'
        unique_states = df_coefs['game_state'].unique()
        
        # A. League Wide Scatter (Offense vs Defense Intercepts)
        for state in unique_states:
            df_plot = df_coefs[mask_intercept & (df_coefs['game_state'] == state)]
            if df_plot.empty:
                continue
                
            df_pivot = df_plot.pivot(index='team', columns='role', values='coef')
            
            if 'Offense' in df_pivot.columns and 'Defense' in df_pivot.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                sns.scatterplot(data=df_pivot, x='Offense', y='Defense', ax=ax)
                
                # Improved label placement — use per-point offsets and smaller font
                for team, row in df_pivot.iterrows():
                    ax.annotate(team, (row['Offense'], row['Defense']),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=8, alpha=0.85)
                    
                ax.set_title(f"Team Strength: {state} (Intercepts)\n"
                             f"Positive Offense = Good | Negative Defense = Good")
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                
                # Negative Defense coef = "lowers xG against" = good defense
                ax.invert_yaxis()
                ax.set_ylabel("Defensive Impact (Lower is Better)")
                ax.set_xlabel("Offensive Impact (Higher is Better)")
                
                fig.tight_layout()
                fig.savefig(output_dir / f"scatter_intercepts_{state}.png", dpi=150)
                plt.close(fig)
                    
        # B. Top 10 Bars per State/Role
        for state in unique_states:
            for role in ['Offense', 'Defense']:
                df_sub = df_coefs[(df_coefs['game_state'] == state) & 
                                  (df_coefs['role'] == role) & 
                                  (mask_intercept)]
                                  
                if df_sub.empty:
                    continue
                    
                ascending = True if role == 'Defense' else False
                df_sorted = df_sub.sort_values('coef', ascending=ascending).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=df_sorted, x='coef', y='team', hue='team',
                            palette='viridis', legend=False, ax=ax)
                ax.set_title(f"Top 10 {role} ({state})")
                ax.set_xlabel("Coefficient Impact")
                fig.tight_layout()
                fig.savefig(output_dir / f"top10_{role}_{state}.png", dpi=150)
                plt.close(fig)

        # C. Team Specific Plots
        teams_dir = output_dir / "teams"
        teams_dir.mkdir(exist_ok=True)
        
        unique_teams = df_coefs['team'].unique()
        
        if teams_filter:
            unique_teams = [t for t in unique_teams if t in teams_filter]
            logger.info(f"Filtering to {len(unique_teams)} teams: {unique_teams}")
        
        logger.info(f"Generating individual plots for {len(unique_teams)} teams...")
        
        for team in unique_teams:
            df_team = df_coefs[df_coefs['team'] == team]
            
            # Intercepts (Overall Strength)
            df_int = df_team[df_team['feature'] == 'intercept']
            if not df_int.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=df_int, x='game_state', y='coef', hue='role', ax=ax)
                ax.set_title(f"{team} - Overall Adjustments (Intercepts)")
                ax.axhline(0, color='k', linewidth=0.5)
                ax.set_ylabel("Impact on Log-Odds")
                fig.tight_layout()
                fig.savefig(teams_dir / f"{team}_intercepts.png", dpi=150)
                plt.close(fig)
