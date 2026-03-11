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
                           verbose: bool = True):
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
    
    dashboard_scripts = [
        ('nested_model_dashboard.py', 'analysis/nested_model_dashboard.html'),
        # ('generate_blocked_shot_debug_dashboard.py', 'analysis/blocked_shot_debug.html'),  # Optional
    ]
    
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
                    vprint(f"    ✓ {output_name}")
                else:
                    vprint(f"    ✗ Failed: {result.stderr[:200]}")
            except Exception as e:
                vprint(f"    ✗ Error: {e}")
        else:
            vprint(f"  Skipping {script_name} (not found)")
    
    # 3. Text Summary
    vprint("\n[3/5] Generating text summary...")
    summary_path = Path(output_dir) / 'model_summary.txt'
    
    try:
        _generate_text_summary(model, summary_path, model_path)
        artifacts['summary'] = str(summary_path)
        vprint(f"  ✓ {summary_path}")
    except Exception as e:
        vprint(f"  ✗ Error: {e}")
    
    # 4. Feature Analysis
    vprint("\n[4/5] Feature analysis...")
    feature_path = Path(output_dir) / 'feature_analysis.txt'
    
    try:
        _generate_feature_analysis(model, feature_path)
        artifacts['feature_analysis'] = str(feature_path)
        vprint(f"  ✓ {feature_path}")
    except Exception as e:
        vprint(f"  ✗ Error: {e}")
    
    # 5. Calibration Testing
    vprint("\n[5/5] Calibration testing...")
    calibration_script = scripts_dir / 'verify_calibration.py'
    calibration_output = Path(output_dir) / 'calibration_results.txt'
    
    if calibration_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(calibration_script), '--model', model_path],
                capture_output=True, text=True, timeout=300
            )
            with open(calibration_output, 'w') as f:
                f.write(result.stdout)
            artifacts['calibration'] = str(calibration_output)
            vprint(f"  ✓ {calibration_output}")
            
            # Extract key metrics for display
            if 'Ratio (xG/Goals):' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Ratio' in line or 'Total' in line:
                        vprint(f"    {line.strip()}")
        except Exception as e:
            vprint(f"  ✗ Error: {e}")
    
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
                clf = submodel.named_steps.get('clf')
                preprocessor = submodel.named_steps.get('preprocessor')
                
                if clf and hasattr(clf, 'coef_'):
                    coefs = clf.coef_.flatten()
                    
                    # Try to get feature names
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = [f"feat_{i}" for i in range(len(coefs))]
                    
                    # Sort by absolute magnitude
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    
                    f.write(f"Top 20 Features (by |coefficient|):\n\n")
                    for i, idx in enumerate(sorted_idx[:20]):
                        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                        coef = coefs[idx]
                        f.write(f"  {i+1:2d}. {fname[:40]:<40} {coef:+.4f}\n")
                    
                    f.write(f"\nTotal features: {len(coefs)}\n")
                    f.write(f"Non-zero: {np.sum(coefs != 0)}\n")
                    f.write(f"Intercept: {clf.intercept_[0]:.4f}\n")
                    
            except Exception as e:
                f.write(f"  Error: {e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    # Allow running as script
    generate_model_summary(verbose=True)
