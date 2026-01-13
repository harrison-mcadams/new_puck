
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgboost_nested, fit_xgs, features as feature_util, data_pipeline

# Configure Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NestedGLM(BaseEstimator, ClassifierMixin):
    def __init__(self, features, poly_degree=2):
        self.features = features
        self.poly_degree = poly_degree
        self.model_block = None
        self.model_acc = None
        self.model_finish = None
        
    def _build_pipeline(self):
        # Identify numeric vs categorical
        # Hardcoded based on known feature lists for safety in this script
        cat_features = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state']
        num_features = [f for f in self.features if f not in cat_features]
        
        # Numeric Trans: Impute -> Poly -> Scale
        num_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
            ('scaler', StandardScaler())
        ])
        
        # Cat Trans: Impute -> OHE
        cat_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_trans, num_features),
            ('cat', cat_trans, [f for f in cat_features if f in self.features])
        ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000))
        ])

    def fit(self, X, y=None):
        logger.info("Fitting GLM Block Model...")
        self.model_block = self._build_pipeline()
        y_block = (X['event'] == 'blocked-shot').astype(int)
        self.model_block.fit(X[self.features], y_block)
        
        logger.info("Fitting GLM Accuracy Model...")
        self.model_acc = self._build_pipeline()
        mask_unblocked = X['event'] != 'blocked-shot'
        X_unblocked = X[mask_unblocked]
        y_acc = X_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        self.model_acc.fit(X_unblocked[self.features], y_acc)
        
        logger.info("Fitting GLM Finish Model...")
        self.model_finish = self._build_pipeline()
        mask_on_net = X['event'].isin(['shot-on-goal', 'goal'])
        X_on_net = X[mask_on_net]
        y_finish = (X_on_net['event'] == 'goal').astype(int)
        self.model_finish.fit(X_on_net[self.features], y_finish)
        
        return self

    def predict_proba(self, X):
        # 1. Block
        p_blocked = self.model_block.predict_proba(X[self.features])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        # 2. Acc
        p_on_net = self.model_acc.predict_proba(X[self.features])[:, 1]
        
        # 3. Finish
        p_finish = self.model_finish.predict_proba(X[self.features])[:, 1]
        
        # Combine
        p_goal = p_unblocked * p_on_net * p_finish
        return np.column_stack((1-p_goal, p_goal))

def main():
    print("--- Loading Data ---")
    df = fit_xgs.load_all_seasons_data(base_dir='data')
    if len(df) == 0:
        df = fit_xgs.load_data() # fallback
    
    # Preprocess (Imputation/Adjustment)
    df = data_pipeline.preprocess_features(df, is_training=True, apply_imputation=True, apply_arena_adjustments=True)
    
    # Define Features
    features = feature_util.get_features('all_inclusive')
    print(f"Features: {len(features)}")
    
    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_test = (df_test['event'] == 'goal').astype(int)
    
    # 1. Train XGBoost (Standard)
    print("\n--- Training XGBoost (Baseline) ---")
    xgb_model = fit_xgboost_nested.XGBNestedXGClassifier(features=features, enable_categorical=True)
    xgb_model.fit(df_train)
    p_xgb = xgb_model.predict_proba(df_test)[:, 1]
    
    # 2. Train GLM (Poly)
    print("\n--- Training GLM (Poly degree=2) ---")
    glm_model = NestedGLM(features=features, poly_degree=3) # Degree 3 to capture curves well
    glm_model.fit(df_train)
    p_glm = glm_model.predict_proba(df_test)[:, 1]
    
    # 3. Compare
    print("\n=============================================")
    print("          MODEL COMPARISON REPORT")
    print("=============================================")
    
    # Metrics
    auc_xgb = roc_auc_score(y_test, p_xgb)
    auc_glm = roc_auc_score(y_test, p_glm)
    print(f"AUC (Global):   XGB={auc_xgb:.4f}  |  GLM={auc_glm:.4f}")
    
    ll_xgb = log_loss(y_test, p_xgb)
    ll_glm = log_loss(y_test, p_glm)
    print(f"LogLoss:        XGB={ll_xgb:.4f}  |  GLM={ll_glm:.4f}")
    
    # High Danger Ceiling
    print("\n--- High Danger Analysis (Max xG) ---")
    print(f"Max xG (XGB): {p_xgb.max():.4f}")
    print(f"Max xG (GLM): {p_glm.max():.4f}")
    
    # Top 1% Stats
    thresh_xgb = np.percentile(p_xgb, 99)
    thresh_glm = np.percentile(p_glm, 99)
    mean_top_xgb = p_xgb[p_xgb >= thresh_xgb].mean()
    mean_top_glm = p_glm[p_glm >= thresh_glm].mean()
    print(f"Mean xG (Top 1%): XGB={mean_top_xgb:.4f} | GLM={mean_top_glm:.4f}")
    
    # Visual check on "High Danger" counts
    print(f"Count > 0.3: XGB={(p_xgb > 0.3).sum()} | GLM={(p_glm > 0.3).sum()}")
    print(f"Count > 0.5: XGB={(p_xgb > 0.5).sum()} | GLM={(p_glm > 0.5).sum()}")
    print(f"Count > 0.7: XGB={(p_xgb > 0.7).sum()} | GLM={(p_glm > 0.7).sum()}")
    
    # Plot Calibration
    plt.figure(figsize=(10, 6))
    
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, p_xgb, n_bins=10)
    plt.plot(prob_pred_xgb, prob_true_xgb, marker='o', label=f'XGBoost (Max={p_xgb.max():.2f})')
    
    prob_true_glm, prob_pred_glm = calibration_curve(y_test, p_glm, n_bins=10)
    plt.plot(prob_pred_glm, prob_true_glm, marker='s', label=f'GLM Poly-3 (Max={p_glm.max():.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Goal Rate')
    plt.title('Calibration Comparison: XGBoost vs Poly-GLM')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/comparison_xgb_glm.png')
    print("\nSaved calibration plot to analysis/comparison_xgb_glm.png")

if __name__ == "__main__":
    main()
