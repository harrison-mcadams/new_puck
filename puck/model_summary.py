"""model_summary.py

Consolidated model performance summary generation.
Call `generate_model_summary()` after training to produce all analysis artifacts.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)

# Ensure project root in path
try:
    from . import config as puck_config
except ImportError:
    try:
        import config as puck_config
    except ImportError:
        # Fallback if config is missing
        class DummyConfig:
            ANALYSIS_DIR = 'analysis'
        puck_config = DummyConfig()


def generate_model_summary(model_path: str = None, 
                           test_df: pd.DataFrame = None,
                           output_dir: str = None,
                           verbose: bool = True,
                           **kwargs):
    """
    Generate comprehensive model performance summary after training.
    
    This function:
    1. Regenerates dashboards (nested_model, shot_density)
    2. Creates a text summary file with model/submodel performance
    3. Performs feature importance analysis
    4. Runs calibration testing
    
    Args:
        model_path: Path to the trained model. Defaults to analysis/xgs/xg_model_nested.joblib.
        test_df: Optional test DataFrame for evaluation. If None, loads from training script output.
        output_dir: Directory to save summaries. Defaults to analysis/nested_xgs/.
        verbose: Print progress.
    
    Returns:
        dict with paths to generated artifacts.
    """
    if model_path is None:
        model_path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_model_nested.joblib')
    
    if output_dir is None:
        output_dir = os.path.join(puck_config.ANALYSIS_DIR, 'nested_xgs')
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    
    def vprint(*args):
        if verbose:
            print(*args)
    
    vprint("\n" + "="*60)
    vprint("GENERATING MODEL SUMMARY")
    vprint("="*60)
    
    # 1. Load Model
    vprint("\n[1/5] Loading model...")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return artifacts
    
    model = joblib.load(model_path)
    vprint(f"  Loaded {type(model).__name__}")
    
    # 2. Regenerate Dashboards
    vprint("\n[2/5] Regenerating dashboards...")
    scripts_dir = Path(__file__).parent.parent / 'scripts'
    
    model_type = type(model).__name__
    dashboard_scripts = []
    
    if model_type == 'XGBNestedXGClassifier':
        dashboard_scripts.append(('xgboost_nested_model_dashboard.py', f'analysis/xgboost_nested_xgs/{Path(model_path).stem}_dashboard.html'))
    elif model_type == 'XGBAlternateXGClassifier':
        dashboard_scripts.append(('xgboost_alternate_model_dashboard.py', f'analysis/xgboost_alternate_xgs/{Path(model_path).stem}_dashboard.html'))
    elif model_type == 'XGBNonNestedXGClassifier':
        dashboard_scripts.append(('xgboost_non_nested_model_dashboard.py', f'analysis/xgboost_non_nested_xgs/{Path(model_path).stem}_dashboard.html'))
    elif model_type == 'NestedGLM':
        dashboard_scripts.append(('nested_model_dashboard.py', 'analysis/nested_model_dashboard.html'))
    elif model_type == 'NonNestedGLM':
        dashboard_scripts.append(('non_nested_model_dashboard.py', 'analysis/non_nested_model_dashboard.html'))
    elif model_type == 'XGBTensorXGClassifier':
        dashboard_scripts.append(('xgboost_tensor_model_dashboard.py', f'analysis/xgboost_tensor_xgs_modern/{Path(model_path).stem}_dashboard.html'))
    else:
        vprint(f"  Warning: Unknown model type {model_type}. No dashboard script assigned.")
    
    for script_name, output_name in dashboard_scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            vprint(f"  Running {script_name}...")
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), model_path],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    artifacts[script_name] = output_name
                    vprint(f"    [OK] {output_name}")
                else:
                    vprint(f"    [FAILED] {result.stderr[:200]}")
            except Exception as e:
                vprint(f"    [ERROR] {e}")
        else:
            vprint(f"  Skipping {script_name} (not found)")
    
    # 3. Text Summary
    vprint("\n[3/5] Generating text summary...")
    summary_path = Path(output_dir) / 'model_summary.txt'
    
    try:
        _generate_text_summary(model, summary_path, model_path)
        artifacts['summary'] = str(summary_path)
        vprint(f"  [OK] {summary_path}")
    except Exception as e:
        vprint(f"  [ERROR] {e}")
    
    # 4. Feature Analysis
    vprint("\n[4/5] Feature analysis...")
    feature_path = Path(output_dir) / 'feature_analysis.txt'
    
    try:
        _generate_feature_analysis(model, feature_path)
        artifacts['feature_analysis'] = str(feature_path)
        vprint(f"  [OK] {feature_path}")
    except Exception as e:
        vprint(f"  [ERROR] {e}")
    
    # 5. Calibration Testing
    vprint("\n[5/6] Calibration testing...")
    calibration_output = Path(output_dir) / 'calibration_results.txt'
    
    if test_df is not None:
        try:
            _generate_calibration_results(model, test_df, calibration_output)
            artifacts['calibration'] = str(calibration_output)
            vprint(f"  [OK] {calibration_output}")
        except Exception as e:
            vprint(f"  [ERROR] {e}")
    else:
        vprint("  Skipping Calibration (no test_df provided)")
    
    # 6. Predictive Power Analysis (Optional)
    vprint("\n[6/6] Predictive power analysis...")
    if kwargs.get('run_predictive_analysis', False):
        predictive_script = scripts_dir / 'evaluate_predictive_power.py'
        if predictive_script.exists():
            vprint("  Running comprehensive predictive power comparison...")
            try:
                # Use the user's preferred "Boss Command" parameters
                cmd = [
                    sys.executable, str(predictive_script),
                    "--seasons", "20202021+",
                    "--model", "xgboost_nested,xgboost_non_nested,xgboost_alternate,nested,non_nested,actual",
                    "--filter", "all",
                    "--n-boot", "100",
                    "--parallel",
                    "--n-jobs", "-1"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                
                predictive_output = Path(output_dir) / 'predictive_power_results.txt'
                with open(predictive_output, 'w') as f:
                    f.write(result.stdout)
                
                artifacts['predictive_power'] = str(predictive_output)
                vprint(f"  [OK] {predictive_output}")
                
                # Display top-level result if found
                for line in result.stdout.split('\n'):
                    if 'Stability' in line or 'MAE' in line:
                        vprint(f"    {line.strip()}")
            except Exception as e:
                vprint(f"  [ERROR] {e}")
        else:
            vprint("  Skipping Predictive Analysis (script not found)")
    else:
        vprint("  Skipping Predictive Analysis (run_predictive_analysis=False)")

    vprint("\n" + "="*60)
    vprint("SUMMARY COMPLETE")
    vprint("="*60)
    vprint(f"Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        vprint(f"  - {name}: {path}")
    
    return artifacts


def _generate_text_summary(model, output_path: Path, model_path: str):
    """Generate text file with model performance characteristics."""
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NESTED GLM MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Model Info
        f.write("## Model Configuration\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        
        if hasattr(model, 'use_splines'):
            f.write(f"Splines Enabled: {model.use_splines}\n")
        if hasattr(model, 'poly_degree'):
            f.write(f"Polynomial Degree: {model.poly_degree}\n")
        if hasattr(model, 'n_estimators'):
            f.write(f"XGBoost Estimators: {model.n_estimators}\n")
        if hasattr(model, 'max_depth'):
            f.write(f"XGBoost Max Depth: {model.max_depth}\n")
        if hasattr(model, 'enable_marginalization'):
            f.write(f"Marginalization: {model.enable_marginalization}\n")
        
        # Features
        f.write(f"\n## Features ({len(model.features)})\n")
        for feat in model.features:
            f.write(f"  - {feat}\n")
        
        # Shot Type Priors
        if hasattr(model, 'shot_type_priors_') and model.shot_type_priors_:
            f.write("\n## Shot Type Priors (Marginalization Weights)\n")
            for st, prob in sorted(model.shot_type_priors_.items(), key=lambda x: -x[1]):
                f.write(f"  {st}: {prob:.1%}\n")
        
        # Sub-Model Info
        f.write("\n## Sub-Models\n")
        for name, submodel_attr in [('Block', 'model_block'), ('Accuracy', 'model_acc'), ('Finish', 'model_finish')]:
            submodel = getattr(model, submodel_attr, None)
            if submodel:
                try:
                    clf = submodel.named_steps.get('clf')
                    if clf:
                        f.write(f"\n### {name} Model\n")
                        f.write(f"  Classifier: {type(clf).__name__}\n")
                        if hasattr(clf, 'C'):
                            f.write(f"  Regularization (C): {clf.C}\n")
                        if hasattr(clf, 'n_features_in_'):
                            f.write(f"  Input Features: {clf.n_features_in_}\n")
                        if hasattr(clf, 'coef_'):
                            f.write(f"  Coefficients: {clf.coef_.shape}\n")
                except Exception as e:
                    f.write(f"  Error reading {name}: {e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def _generate_feature_analysis(model, output_path: Path):
    """Analyze feature coefficients from each sub-model."""
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FEATURE COEFFICIENT ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        for name, submodel_attr in [('Block', 'model_block'), ('Accuracy', 'model_acc'), ('Finish', 'model_finish')]:
            submodel = getattr(model, submodel_attr, None)
            if not submodel:
                continue
                
            f.write(f"\n## {name} Model\n")
            f.write("-"*40 + "\n")
            
            try:
                # GLM handling
                clf = None
                preprocessor = None
                if hasattr(submodel, 'named_steps'):
                    clf = submodel.named_steps.get('clf')
                    preprocessor = submodel.named_steps.get('preprocessor')
                
                # XGBoost handling (directly a model or has internal models)
                if not clf:
                    clf = submodel # Could be the XGBClassifier itself
                
                if clf and hasattr(clf, 'coef_'):
                    coefs = clf.coef_.flatten()
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = [f"feat_{i}" for i in range(len(coefs))]
                    
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    f.write(f"Top 20 Features (by |coefficient|):\n\n")
                    for i, idx in enumerate(sorted_idx[:20]):
                        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                        f.write(f"  {i+1:2d}. {fname[:40]:<40} {coefs[idx]:+.4f}\n")
                        
                elif clf and hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    
                    # Priority 1: Check if the classifier has its own feature names (XGBoost >= 1.5)
                    feature_names = getattr(clf, 'feature_names_in_', None)
                    
                    # Priority 2: Check model-specific feature lists (Nested models)
                    if feature_names is None:
                        attr_map = {'Block': 'features_block', 'Accuracy': 'features_acc', 'Finish': 'features_fin'}
                        feat_attr = attr_map.get(name)
                        if feat_attr and hasattr(model, feat_attr):
                            feature_names = getattr(model, feat_attr)
                    
                    # Priority 3: Fallback to global model features
                    if feature_names is None or len(feature_names) != len(importances):
                        feature_names = getattr(model, 'features', None)
                        
                    # Final Fallback
                    if feature_names is None or len(feature_names) != len(importances):
                        feature_names = [f"feat_{i}" for i in range(len(importances))]
                        
                    sorted_idx = np.argsort(importances)[::-1]
                    f.write(f"Top 20 Features (by Importance):\n\n")
                    for i, idx in enumerate(sorted_idx[:20]):
                        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                        f.write(f"  {i+1:2d}. {fname[:40]:<40} {importances[idx]:.4f}\n")
                    
            except Exception as e:
                f.write(f"  Error: {e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def _generate_calibration_results(model, test_df: pd.DataFrame, output_path: Path):
    """Calculate and save calibration metrics for overall and submodels."""
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CALIBRATION & PERFORMANCE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        # 1. Overall xG Calibration
        y_true = (test_df['event'] == 'goal').astype(int)
        try:
            # predict_proba returns [P(no goal), P(goal)]
            probs = model.predict_proba(test_df)[:, 1]
            
            total_xg = probs.sum()
            total_goals = y_true.sum()
            ratio = total_xg / total_goals if total_goals > 0 else 0
            
            auc = roc_auc_score(y_true, probs)
            ll = log_loss(y_true, probs)
            brier = brier_score_loss(y_true, probs)
            
            f.write("## Overall Goal Prediction (xG)\n")
            f.write(f"  Total Shots:        {len(test_df)}\n")
            f.write(f"  Total Actual Goals: {total_goals}\n")
            f.write(f"  Total Predicted xG: {total_xg:.2f}\n")
            f.write(f"  Ratio (xG/Goals):   {ratio:.4f}\n")
            f.write(f"  AUC:                {auc:.4f}\n")
            f.write(f"  LogLoss:            {ll:.4f}\n")
            f.write(f"  Brier Score:        {brier:.6f}\n\n")
        except Exception as e:
            f.write(f"## Overall Goal Prediction\n  Error: {e}\n\n")

        # 2. Submodel Calibration (if Nested)
        if hasattr(model, 'predict_proba_layer'):
            f.write("## Submodel Performance Breakdown\n")
            f.write("-" * 40 + "\n")
            
            # --- Block Layer ---
            try:
                y_block = (test_df['event'] == 'blocked-shot').astype(int)
                p_block = model.predict_proba_layer(test_df, 'block')
                # Ensure 1D
                if len(p_block.shape) > 1: p_block = p_block[:, 1]
                
                f.write("\n### Block Layer (P(Blocked | Shot))\n")
                f.write(f"  AUC:         {roc_auc_score(y_block, p_block):.4f}\n")
                f.write(f"  LogLoss:     {log_loss(y_block, p_block):.4f}\n")
                f.write(f"  Brier Score: {brier_score_loss(y_block, p_block):.6f}\n")
                f.write(f"  Actual Rate: {y_block.mean():.2%}\n")
                f.write(f"  Pred Rate:   {p_block.mean():.2%}\n")
            except Exception as e:
                f.write(f"\n### Block Layer\n  Error: {e}\n")
                
            # --- Accuracy Layer ---
            try:
                mask_unblocked = (test_df['event'] != 'blocked-shot')
                df_unblocked = test_df[mask_unblocked]
                y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
                p_acc = model.predict_proba_layer(df_unblocked, 'accuracy')
                if len(p_acc.shape) > 1: p_acc = p_acc[:, 1]
                
                f.write("\n### Accuracy Layer (P(On Net | Unblocked))\n")
                f.write(f"  AUC:         {roc_auc_score(y_acc, p_acc):.4f}\n")
                f.write(f"  LogLoss:     {log_loss(y_acc, p_acc):.4f}\n")
                f.write(f"  Brier Score: {brier_score_loss(y_acc, p_acc):.6f}\n")
                f.write(f"  Actual Rate: {y_acc.mean():.2%}\n")
                f.write(f"  Pred Rate:   {p_acc.mean():.2%}\n")
            except Exception as e:
                f.write(f"\n### Accuracy Layer\n  Error: {e}\n")
                
            # --- Finish Layer ---
            try:
                mask_on_net = test_df['event'].isin(['shot-on-goal', 'goal'])
                df_on_net = test_df[mask_on_net]
                y_fin = (df_on_net['event'] == 'goal').astype(int)
                p_fin = model.predict_proba_layer(df_on_net, 'finish')
                if len(p_fin.shape) > 1: p_fin = p_fin[:, 1]
                
                f.write("\n### Finish Layer (P(Goal | On Net))\n")
                f.write(f"  AUC:         {roc_auc_score(y_fin, p_fin):.4f}\n")
                f.write(f"  LogLoss:     {log_loss(y_fin, p_fin):.4f}\n")
                f.write(f"  Brier Score: {brier_score_loss(y_fin, p_fin):.6f}\n")
                f.write(f"  Actual Rate: {y_fin.mean():.2%}\n")
                f.write(f"  Pred Rate:   {p_fin.mean():.2%}\n")
            except Exception as e:
                f.write(f"\n### Finish Layer\n  Error: {e}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    # Allow running as script
    generate_model_summary(verbose=True)
