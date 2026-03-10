"""nested_model_dashboard.py

Generates an interactive HTML dashboard for the NestedGLM tensor-product xG model.

KNOWN PITFALLS (these have caused bugs before — read before modifying):

1. BINARY FEATURES: If a new binary (0/1) feature is added to the model
   (e.g. is_home, is_rush, is_rebound), it MUST be routed through the
   'binary' transformer group in fit_glm_nested._build_pipeline — NOT
   through SplineTransformer. A binary feature expanded into 7 spline
   basis functions shifts ALL subsequent coefficient indices and breaks
   the dashboard's spatial heatmaps.

2. SPATIAL GRID EXTRAPOLATION: The rink grid evaluates the tensor spline
   at (distance, angle) combos far outside the training data range (e.g.
   behind center ice). The tensor product of B-spline bases can produce
   extreme values (100+) at these points. compute_spatial_grid() must
   clamp scores to a reasonable range (currently P1/P99).

3. JS SPLINE BASIS TRIMMING: sklearn's SplineTransformer(include_bias=False)
   drops the LAST basis function. The JS bspline_basis() produces all N+1
   functions. When trimming to match, we must keep the FIRST n elements:
       basis.slice(0, n)     # CORRECT — matches sklearn
       basis.slice(n_extra)  # WRONG — phase-shifts all evaluations
   Getting this wrong causes every spline feature to be evaluated with
   misaligned coefficients, producing wildly incorrect predictions.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import warnings
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder, PolynomialFeatures

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, fit_glm_nested, fit_xgs, features as feature_util, rink

import math
try:
    from puck.rink import calculate_distance_and_angle
except ImportError:
    # Fallback
    def calculate_distance_and_angle(x, y, goal_x, goal_y=0.0):
        # Simplified copy
        distance = math.hypot(x - goal_x, y - goal_y)
        vx = x - goal_x
        vy = y - goal_y
        if goal_x < 0: rx, ry = 0.0, 1.0
        else: rx, ry = 0.0, -1.0
        cross = rx * vy - ry * vx
        dot = rx * vx + ry * vy
        angle_rad_ccw = math.atan2(cross, dot)
        angle_deg = (-math.degrees(angle_rad_ccw)) % 360.0
        return distance, angle_deg

def compute_spatial_grid(pipeline, feature_names):
    """
    Computes the spatial component score for the standard 50x43 grid.
    Returns a flattened list or 2D list of scores.
    """
    # Grid definition (Matches JS)
    X_POINTS = 50
    Y_POINTS = 43
    xs = np.linspace(0, 100, X_POINTS)
    ys = np.linspace(-42.5, 42.5, Y_POINTS)
    
    # Generate points
    # We need to simulate the pipeline for just the spatial features.
    # 1. Identify spatial coefs
    clf = pipeline.named_steps['clf']
    pre = pipeline.named_steps['preprocessor']
    
    try:
        # Get feature names out to align coefs
        # Note: NestedGLM pipelines might be intricate.
        # Let's rely on the transformer name 'spatial_tensor' if it exists.
        
        # We need to run the transformation on the grid points relative to the 'spatial_tensor' step.
        # Then multiply by the corresponding coefficients.
        
        # Find the spatial_tensor step in ColumnTransformer
        tensor_transformer = None
        for name, trans, cols in pre.transformers_:
            if name == 'spatial_tensor':
                tensor_transformer = trans
                break
        
        if tensor_transformer is None:
            return None # No spatial tensor
            
        # Extract indices of spatial features in the final feature vector
        all_out_feats = pre.get_feature_names_out()
        spatial_indices = [i for i, f in enumerate(all_out_feats) if f.startswith('spatial_tensor__')]
        
        if not spatial_indices:
            return None

        spatial_coefs = clf.coef_[0][spatial_indices]
        # Note: We DON'T include intercept in the "Spatial Grid" usually, 
        # or we do and then later add non-spatial parts.
        # Let's include ONLY the spatial contribution. Intercept is handled in JS base.
        
        # Generate Grid Data
        grid_rows = []
        for y in ys:
            for x in xs:
                # Goal at 89, 0 assumption (Standard)
                dist, ang = calculate_distance_and_angle(x, y, 89.0, 0.0)
                grid_rows.append({'distance': dist, 'angle_deg': ang})
        
        grid_df = pd.DataFrame(grid_rows)
        
        # Transform
        X_trans = tensor_transformer.transform(grid_df)
        
        # Calculate Scores
        scores = X_trans @ spatial_coefs
        
        # PITFALL #2: Spatial grid extrapolation blow-up.
        # The full rink grid includes (distance, angle) pairs that never
        # appear in training data (e.g. 90ft away at 0° angle).  The
        # tensor product of B-spline bases can produce extreme scores
        # (100+) at these points, saturating the heatmap.  Clamp to
        # P1/P99 to suppress outliers while preserving the valid range.
        p1, p99 = np.percentile(scores, [1, 99])
        scores = np.clip(scores, p1, p99)
        
        # Reshape to 2D list [row][col] -> [y][x]
        # Our loop was y outer, x inner
        grid_2d = []
        idx = 0
        for _ in ys:
            row = []
            for _ in xs:
                row.append(float(scores[idx]))
                idx += 1
            grid_2d.append(row)
            
        return grid_2d

    except Exception as e:
        print(f"Error computing spatial grid: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_rink_shapes(xref='x', yref='y'):
    """Full rink shapes for offensive zone."""
    shapes = []
    line_color = "rgba(0, 0, 0, 0.3)"
    red_line_color = "rgba(255, 0, 0, 0.3)"
    blue_line_color = "rgba(0, 0, 255, 0.3)"
    
    shapes.append(dict(type="line", x0=0, y0=42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=100, y1=-42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=100, y0=-42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=0, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=2)))
    shapes.append(dict(type="line", x0=25, y0=-42.5, x1=25, y1=42.5, xref=xref, yref=yref, line=dict(color=blue_line_color, width=2)))
    shapes.append(dict(type="line", x0=89, y0=-42.5, x1=89, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=1)))
    shapes.append(dict(type="circle", x0=89-4, y0=-4, x1=89+4, y1=4, xref=xref, yref=yref, line=dict(color=red_line_color, width=1)))
    return shapes

def load_data_and_priors(features):
    """Loads a sample of data to compute priors for all categorical features."""
    print("Loading data for priors...")
    try:
        # Load a subset for speed? No, full for accuracy
        df = fit_xgs.load_data()
        
        priors = {}
        # Identify categorical features
        # We don't have the config here easily, try to infer or be broad
        # Or iterate over all features
        for f in features:
            if f in df.columns:
                # Check if seemingly categorical (string or few num values)
                if df[f].dtype == object or df[f].nunique() < 20: 
                    # If numeric low cardinality (like rush=0/1), treat as categorical prior
                    # If numeric high cardinality (distance), skip
                    if df[f].dtype != object and df[f].nunique() > 10:
                        continue
                        
                    series = df[f].astype(str)
                    counts = series.value_counts(normalize=True).to_dict()
                    priors[f] = counts
            
        print(f"Computed priors for {len(priors)} features.")
        return priors
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def extract_pipeline_params(pipeline, features):
    """Extracts weights, scales, knots, etc from a single layer pipeline."""
    
    # 1. Pipeline Steps
    # Structure: preprocessor (ColumnTransformer) -> clf (LogisticRegression)
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    
    # 2. Extract Coefficients
    coef = clf.coef_[0].tolist()
    intercept = clf.intercept_[0]
    
    # 3. Extract Transformers
    transformers = {}
    
    current_coef_idx = 0
    spatial_data = None
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder': continue
        
        # Calculate output size to advance coef index
        n_out = 0
        if hasattr(trans, 'get_feature_names_out'):
             try:
                 # Try with input cols if needed (sklearn version dependent)
                 # But trans is usually a pipeline, so no args needed if fitted
                 out_names = trans.get_feature_names_out()
             except:
                 try:
                    out_names = trans.get_feature_names_out(cols)
                 except: 
                    # Fallback for old sklearn or complex pipelines
                     out_names = [] # Dangerous
             n_out = len(out_names)
        elif hasattr(trans, 'categories_'): # OHE direct
             n_out = sum(len(c) for c in trans.categories_)
        
        # If we failed to get size, we might desync. 
        # But specifically for our known pipeline structure:
        
        if name == 'spatial_tensor':
             # We SKIP extracting parameters for JS.
             # Instead we compute the grid!
             # We assume compute_spatial_grid handles finding this step.
             current_coef_idx += n_out
             continue
             
        # Slice Coefs for this block
        block_coefs = coef[current_coef_idx : current_coef_idx + n_out]
        
        # Identify type
        # Check for SplineTransformer vs OHE
        is_spline = False
        is_ohe = False
        step_obj = None
        
        if hasattr(trans, 'steps'):
             for _, step in trans.steps:
                 if isinstance(step, SplineTransformer):
                     is_spline = True
                     step_obj = step
                 elif isinstance(step, OneHotEncoder):
                     is_ohe = True
                     step_obj = step
                     
        if is_spline:
             # Independent Splines (num_spline)
             # trans is Pipeline([imputer, spline, scaler])
             scaler = trans.named_steps['scaler']
             
             feats_data = {}
             # n_out is total. n_per_col = n_out / len(cols)
             if len(cols) > 0:
                 n_per_col = n_out // len(cols)
                 local_idx = 0
                 
                 for i, col in enumerate(cols):
                     # Coefs
                     col_c = block_coefs[local_idx : local_idx + n_per_col]
                     # Scaler
                     col_m = scaler.mean_[local_idx : local_idx + n_per_col].tolist()
                     col_s = scaler.scale_[local_idx : local_idx + n_per_col].tolist()
                     
                     bs = step_obj.bsplines_[i]
                     
                     feats_data[col] = {
                         'type': 'spline',
                         'knots': bs.t.tolist(),
                         'degree': bs.k,
                         'coefs': col_c,
                         'scaler_mean': col_m,
                         'scaler_scale': col_s
                     }
                     local_idx += n_per_col
             
             transformers['num_spline'] = feats_data
             
        elif is_ohe:
             # Categorical
             feats_data = {}
             local_idx = 0
             for i, col in enumerate(cols):
                 cats = step_obj.categories_[i].tolist()
                 w = {}
                 for c_val in cats:
                     w[str(c_val)] = block_coefs[local_idx]
                     local_idx += 1
                 feats_data[col] = {'type': 'ohe', 'weights': w}
             
             transformers['cat'] = feats_data
             
        elif name == 'binary':
             # Binary features: Pipeline([imputer, scaler]) — one coef per feature
             scaler = trans.named_steps['scaler']
             feats_data = {}
             for i, col in enumerate(cols):
                 feats_data[col] = {
                     'type': 'binary',
                     'coef': block_coefs[i],
                     'scaler_mean': float(scaler.mean_[i]),
                     'scaler_scale': float(scaler.scale_[i])
                 }
             transformers['binary'] = feats_data

        elif name == 'num_poly':
             # Poly fallback
             # ... simplified ...
             pass
             
        current_coef_idx += n_out

    # Compute Spatial Grid if 'spatial_tensor' was in modules (implied by features passed)
    spatial_grid = compute_spatial_grid(pipeline, features)
        
    return {
        'intercept': intercept,
        'transformers': transformers,
        'spatial_grid': spatial_grid
    }

def main():
    # Allow command line arg for model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "analysis/xgs/xg_model_nested.joblib"
        
    # Derive output name from model name
    base_name = os.path.basename(model_path).replace('.joblib', '')
    output_path = f"analysis/nested_xgs/{base_name}_dashboard.html"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    # Hack to allow loading Custom Class from __main__ or module 
    # (Since we defined TensorSpline in fit_glm_nested, standard load should work if imports match)
    model = joblib.load(model_path)
    
    print("Extracting model parameters...")
    
    all_features = model.features
    cat_features_list = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state']
    
    actual_cats = [f for f in cat_features_list if f in all_features]
    actual_nums = [f for f in all_features if f not in actual_cats]

    # Layers
    layers_data = {
        'block': extract_pipeline_params(model.model_block, [f for f in all_features if f != 'shot_type']),
        'accuracy': extract_pipeline_params(model.model_acc, all_features),
        'finish': extract_pipeline_params(model.model_finish, all_features)
    }
    
    # Priors
    priors = load_data_and_priors(all_features)
    
    # Export Data
    export_data = {
        'features': all_features,
        'cat_features': actual_cats,
        'num_features': actual_nums,
        'layers': layers_data,
        'priors': priors,
        'defaults': {
            'period_number': 2,
            'time_elapsed_in_period_s': 600,
            'total_time_elapsed_s': 1800,
            'last_event_time_diff': 10,
            'period_time_type': 'elapsed',
            'home_team_defending_side': 'left',
            'player_name': 'Simulated',
            'angle_change_last_event': 0,
            'speed_from_last_event': 0,
            'dist_from_last_event': 0,
            'score_diff': 0,
            'shooter_role': 'F',
            'game_state': '5v5',
            'shot_type': 'wrist',
            'shoots_catches': 'L',
            'is_rush': 0,
            'is_rebound': 0,
            'is_home': 1,
            'rebound_angle_change': 0,
            'rebound_time_diff': 0,
            'last_event_type': 'faceoff' 
        },
        'options': {
            'shooter_role': ['F', 'D', 'Marginalized'],
            'score_diff': ['-2', '-1', '0', '1', '2'],
            'game_state': ['5v5', '5v4', '4v5', '5v3', '3v5', '4v4', '3v3', '6v5', '5v6', '6v6', 'Marginalized'],
            'shot_type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected', 'wrap-around', 'Marginalized'],
            'shoots_catches': ['L', 'R', 'Marginalized'],
            'is_rush': ['0', '1', 'Marginalized'],
            'is_home': ['0', '1', 'Marginalized'],
            'is_rebound': ['0', '1', 'Marginalized'],
            'last_event_type': sorted(list(priors.get('last_event_type', {}).keys())) + ['Marginalized']
        },
        'use_splines': getattr(model, 'use_splines', False)
    }
    
    json_str = json.dumps(export_data)
    
    print(f"Generating HTML to {output_path}...")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Dashboard - {base_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; background: #111; color: white; font-family: sans-serif; overflow: hidden; }}
        #controls {{ 
            position: absolute; top: 0; left: 0; width: 300px; height: 100vh; 
            background: #222; overflow-y: auto; padding: 10px; box-sizing: border-box; 
            border-right: 1px solid #444;
        }}
        #plot {{ 
            position: absolute; top: 0; left: 300px; right: 0; bottom: 0; 
        }}
        .ctrl-group {{ margin-bottom: 12px; }}
        label {{ display: block; font-size: 0.85em; color: #aaa; margin-bottom: 3px; }}
        select, input {{ 
            width: 100%; background: #333; color: white; border: 1px solid #555; 
            padding: 4px; border-radius: 3px; 
        }}
        .btn {{ 
            width: 48%; padding: 8px; border: none; cursor: pointer; color: white; margin-top: 10px;
        }}
        #loading {{
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            font-size: 2em; background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px;
            z-index: 1000; display: block;
        }}
    </style>
</head>
<body>
    <div id="loading">Loading Engine...</div>
    
    <div id="controls">
        <h3>Model Controls</h3>
        <small>{base_name}</small>
        <div id="inputs"></div>
        <button class="btn" style="background: #28a745" onclick="setBaseline()">Set Baseline</button>
        <button class="btn" style="background: #dc3545" onclick="clearBaseline()">Clear Top</button>
    </div>
    
    <div id="plot"></div>
    <div id="debug_log" style="background: #eee; padding: 10px; margin-top: 20px; font-family: monospace; max-height: 200px; overflow: auto;"></div>

<script>
    const MODEL = {json_str};
    const RINK_SHAPES = {json.dumps(get_rink_shapes())};
    
    // Grid State
    const X_POINTS = 50;
    const Y_POINTS = 43;
    let gridX = [];
    let gridY = [];
    
    // Init Grid
    for(let i=0; i<X_POINTS; i++) gridX.push(i * (100/(X_POINTS-1)));
    for(let i=0; i<Y_POINTS; i++) gridY.push(-42.5 + i * (85/(Y_POINTS-1)));
    
    let baselineData = null;

    // --- MATH ENGINE ---
    
    function sigmoid(z) {{
        return 1 / (1 + Math.exp(-z));
    }}
    
    // BSpline Helper
    function bspline_basis(x, t, k) {{
        // Robust Iterative De Boor Algorithm
        // Returns the full vector of k-th degree basis functions N_i,p(x)
        // t is knots array.
        // x is evaluation point.
        // k is degree.
        
        const n = t.length - k - 1; // Number of basis functions
        let N = new Array(n).fill(0);

        // 1. Find span index 'idx' such that t[idx] <= x < t[idx+1]
        let idx = -1;
        
        // Handle boundaries
        if (x < t[0] || x > t[t.length-1]) return N; // Out of bounds
        
        // Special case: x == max knot (right endpoint)
        // Scikit-learn includes the right endpoint in the last interval.
        if (x === t[t.length - 1]) {{
            idx = t.length - k - 2;
        }} else {{
            // Linear search for span
            for(let i=0; i < t.length - 1; i++) {{
                if (x >= t[i] && x < t[i+1]) {{
                    idx = i;
                    break;
                }}
            }}
        }}

        if (idx === -1) return N;

        // 2. Compute non-zero basis functions
        // "NURBS Book" Algorithm A2.2
        // Compute basis functions N[idx-k] ... N[idx]
        
        let basis = new Array(k + 1).fill(0);
        let left = new Array(k + 1).fill(0);
        let right = new Array(k + 1).fill(0);
        
        basis[0] = 1.0;
        
        for (let j = 1; j <= k; j++) {{
            left[j] = x - t[idx + 1 - j];
            right[j] = t[idx + j] - x;
            let saved = 0.0;
            
            for (let r = 0; r < j; r++) {{
                let term = basis[r] / (right[r + 1] + left[j - r]);
                basis[r] = saved + right[r + 1] * term;
                saved = left[j - r] * term;
            }}
            basis[j] = saved;
        }}
        
        // 3. Scatter into full vector
        // basis[0] corresponds to index (idx - k)
        // basis[k] corresponds to index (idx)
        
        for(let j = 0; j <= k; j++) {{
            let map_idx = idx - k + j;
            if (map_idx >= 0 && map_idx < n) {{
                N[map_idx] = basis[j];
            }}
        }}

        return N;
    }}
    
    function transform_numeric_feature(val, config) {{
        if (config.type === 'spline') {{
            let basis = bspline_basis(val, config.knots, config.degree);
            // PITFALL #3: sklearn's include_bias=False drops the LAST
            // basis function.  We MUST slice from the front (keep first n).
            // Using basis.slice(basis.length - n) drops the FIRST and
            // phase-shifts every coefficient, causing wildly wrong scores.
            if (basis.length > config.scaler_mean.length) {{
                 basis = basis.slice(0, config.scaler_mean.length);
            }}
            
            // Scale
            let out = [];
            for(let i=0; i<basis.length; i++) {{
                out.push( (basis[i] - config.scaler_mean[i]) / config.scaler_scale[i] );
            }}
            return out; // Vector
        }}
        return [0];
    }}

    
    function transform_tensor(inputs, config) {{
        // Obsolete: Spatial Logic moved to Python pre-calculation
        return [];
    }}
    
    function get_layer_score(features_dict, layer_name, grid_r, grid_c) {{
        const layer = MODEL.layers[layer_name];
        if (!layer) return -999;
        
        let score = layer.intercept;
        
        // Transformers
        const trans = layer.transformers;
        
        // 1. Spatial Grid Lookup
        if (layer.spatial_grid) {{
             // grid_r, grid_c correspond to the indices in the mesh
             // We access them by passing them as args
             if (grid_r !== undefined && layer.spatial_grid[grid_r] && layer.spatial_grid[grid_r][grid_c] !== undefined) {{
                 score += layer.spatial_grid[grid_r][grid_c];
             }}
        }}
        
        // 2. Independent Splines
        if (trans.num_spline) {{
             for (const [feat_name, config] of Object.entries(trans.num_spline)) {{
                let val = features_dict[feat_name];
                let vec = transform_numeric_feature(val, config);
                for(let i=0; i<vec.length; i++) {{
                    score += vec[i] * config.coefs[i];
                }}
             }}
        }}
        
        // 3. Poly (Legacy/Fallback)
        if (trans.num_poly) {{
            // ... (Simple poly implementation omitted for brevity unless needed)
            // Assuming we are in Spline mode mainly
        }}

        // 4. Categorical
        if (trans.cat) {{
            for (const [feat_name, config] of Object.entries(trans.cat)) {{
                let val = String(features_dict[feat_name]);
                let w = config.weights[val];
                if (w !== undefined) score += w;
            }}
        }}
        
        // 5. Binary features (simple scaled linear)
        if (trans.binary) {{
            for (const [feat_name, config] of Object.entries(trans.binary)) {{
                let val = parseFloat(features_dict[feat_name]) || 0;
                score += ((val - config.scaler_mean) / config.scaler_scale) * config.coef;
            }}
        }}
        
        return score;
    }}
    
    function predict_scenario(inputs) {{
        // Standard marginalization logic...
        // Copied from previous, preserving structure
        
        let marg_keys = [];
        let fixed_inputs = {{...inputs}};
        
        for (const k in inputs) {{
            if (inputs[k] === 'Marginalized' && MODEL.options[k]) {{
                marg_keys.push(k);
            }}
        }}
        
        for (const k in fixed_inputs) {{
            if (marg_keys.includes(k)) continue;
            if (['Low', 'Med', 'High', 'Marginalized'].includes(fixed_inputs[k])) {{
                fixed_inputs[k] = 0.0; 
            }}
        }}
        
        let scenarios = [{{weight: 1.0, inputs: fixed_inputs}}];
        
        for (const key of marg_keys) {{
            let new_scenarios = [];
            let opts = Object.keys(MODEL.priors[key] || {{}});
            if (opts.length === 0) opts = MODEL.options[key].filter(x => x!=='Marginalized');
            
            for (const scen of scenarios) {{
                for (const opt of opts) {{
                    let p = (MODEL.priors[key] && MODEL.priors[key][opt]) || (1.0/opts.length);
                    let s = {{ weight: scen.weight * p, inputs: {{...scen.inputs}} }};
                    s.inputs[key] = opt;
                    new_scenarios.push(s);
                }}
            }}
            scenarios = new_scenarios;
        }}
        
        scenarios = scenarios.filter(s => s.weight > 0.001);
        
        let H = Y_POINTS;
        let W = X_POINTS;
        let Z_block = new Float32Array(H*W);
        let Z_acc = new Float32Array(H*W);
        let Z_fin = new Float32Array(H*W);
        let Z_xg = new Float32Array(H*W);
        
        let total_weight = 0;
        
        for (const scen of scenarios) {{
            const w = scen.weight;
            total_weight += w;
            
            for(let r=0; r<H; r++) {{
                for(let c=0; c<W; c++) {{
                    const x = gridX[c];
                    const y = gridY[r];
                    const idx = r*W + c;
                    
                    const dist = Math.sqrt((x-89)**2 + y**2);
                    
                    const dx = x - 89;
                    const dy = y;
                    const angle_rad = Math.atan2(dx, -dy);
                    let angle_deg = -angle_rad * 180 / Math.PI;
                    angle_deg = angle_deg % 360;
                    if (angle_deg < 0) angle_deg += 360;
                    
                    let feats = {{...scen.inputs}};
                    feats.distance = dist;
                    feats.angle_deg = angle_deg;
                    // Note: dist_angle interaction is now handled by Tensor logic automatically if used
                    
                    let s_block = get_layer_score(feats, 'block', r, c);
                    let p_block = sigmoid(s_block);
                    let s_acc = get_layer_score(feats, 'accuracy', r, c);
                    let p_acc = sigmoid(s_acc);
                    let s_fin = get_layer_score(feats, 'finish', r, c);
                    let p_fin = sigmoid(s_fin);
                    
                    let p_xg = (1 - p_block) * p_acc * p_fin;
                    
                    Z_block[idx] += p_block * w;
                    Z_acc[idx] += p_acc * w;
                    Z_fin[idx] += p_fin * w;
                    Z_xg[idx] += p_xg * w;
                }}
            }}
        }}
        
        if(total_weight > 0) {{
            for(let i=0; i<Z_block.length; i++) {{
                Z_block[i] /= total_weight;
                Z_acc[i] /= total_weight;
                Z_fin[i] /= total_weight;
                Z_xg[i] /= total_weight;
            }}
        }}
        
        return [Z_block, Z_acc, Z_fin, Z_xg];
    }}

    // --- UI LOGIC ---
    
    function init() {{
        const inputDiv = document.getElementById('inputs');
        
        // Create Dropdowns
        // Use MODEL.features but filtered for usefulness?
        // We iterate MODEL.options keys
        
        // Group parameters
        const groups = {{
            'Context': ['game_state', 'score_diff', 'period_number', 'is_home'],
            'Shooter': ['shooter_role', 'shoots_catches', 'shot_type'],
            'Play Info': ['is_rush', 'is_rebound', 'last_event_type', 'speed_from_last_event']
        }};
        
        // Flatten for generation
        let placed = [];
        
        for(const [gname, fields] of Object.entries(groups)) {{
            let fs = document.createElement('fieldset');
            fs.className = 'ctrl-group';
            fs.innerHTML = `<legend>${{gname}}</legend>`;
            
            fields.forEach(f => {{
                 if (!MODEL.features.includes(f) && !['score_diff'].includes(f)) return; // Skip if not in model
                 placed.push(f);
                 
                 let wrap = document.createElement('div');
                 let lbl = document.createElement('label');
                 lbl.innerText = f;
                 
                 let sel = document.createElement('select');
                 sel.id = 'in_' + f;
                 sel.onchange = updatePlot;
                 
                 // Options
                 let opts = MODEL.options[f] || ['Marginalized', 'Low', 'Med', 'High'];
                 opts.forEach(o => {{
                     let opt = document.createElement('option');
                     opt.value = o;
                     opt.innerText = o;
                     sel.appendChild(opt);
                 }});
                 
                 // Set Default
                 if (MODEL.defaults[f]) sel.value = MODEL.defaults[f];
                 
                 wrap.appendChild(lbl);
                 wrap.appendChild(sel);
                 fs.appendChild(wrap);
            }});
            
            inputDiv.appendChild(fs);
        }}
        
        // Init Plot
        Plotly.newPlot('plot', [], {{
           grid: {{rows: 2, columns: 4, pattern: 'independent'}},
           paper_bgcolor: '#111', plot_bgcolor: '#111',
           width: document.getElementById('plot').clientWidth,
           height: document.getElementById('plot').clientHeight,
           shapes: RINK_SHAPES, // Need to replicate for all subplots or use layout.shapes?
             // Plotly shapes are usually per axis.
             // We can generate them dynamically in JS.
           xaxis: {{range: [0, 100], visible: false}},
           yaxis: {{range: [-42.5, 42.5], visible: false}},
           // Set up 8 axes?
        }});
        
        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }}
    
    function getInputs() {{
        let inp = {{}};
        // Scrape all selects
        const sels = document.querySelectorAll('select');
        sels.forEach(s => {{
            const name = s.id.substring(3); // strip 'in_'
            inp[name] = s.value;
        }});
        
        // Merge defaults for missing
        for (const k in MODEL.defaults) {{
            if (inp[k] === undefined) inp[k] = MODEL.defaults[k];
        }}
        return inp;
    }}
    
    function updatePlot() {{
        // 1. Get Inputs
        const inputs = getInputs();
        
        // 2. Predict
        console.time("Predict");
        const res = predict_scenario(inputs);
        console.timeEnd("Predict");
        
        // Unpack
        const z_blk = convertTo2D(res[0]);
        const z_acc = convertTo2D(res[1]);
        const z_fin = convertTo2D(res[2]);
        const z_xg = convertTo2D(res[3]);
        
        // Deltas
        let d_blk = z_blk, d_acc = z_acc, d_fin = z_fin, d_xg = z_xg;
        
        if (baselineData) {{
            d_blk = calcDelta(z_blk, baselineData[0]);
            d_acc = calcDelta(z_acc, baselineData[1]);
            d_fin = calcDelta(z_fin, baselineData[2]);
            d_xg = calcDelta(z_xg, baselineData[3]);
        }} else {{
            // Show empty/zeros if no baseline? Or just repeat?
            // Let's show zeros
            const zeros = z_blk.map(r => r.map(c => 0));
            d_blk = zeros; d_acc = zeros; d_fin = zeros; d_xg = zeros;
        }}
        
        // 3. Traces
        // Row 1
        const trace1 = {{type: 'heatmap', x: gridX, y: gridY, z: z_blk, colorscale: 'Magma', zmin:0, zmax:1, name:'Block', xaxis:'x', yaxis:'y'}};
        const trace2 = {{type: 'heatmap', x: gridX, y: gridY, z: z_acc, colorscale: 'Viridis', zmin:0, zmax:1, name:'Acc', xaxis:'x2', yaxis:'y2'}};
        const trace3 = {{type: 'heatmap', x: gridX, y: gridY, z: z_fin, colorscale: 'Viridis', zmin:0, zmax:1, name:'Fin', xaxis:'x3', yaxis:'y3'}};
        const trace4 = {{type: 'heatmap', x: gridX, y: gridY, z: z_xg, colorscale: 'Plasma', zmin:0, zmax:0.3, name:'xG', xaxis:'x4', yaxis:'y4'}};
        
        // Row 2 (Deltas)
        const trace5 = {{type: 'heatmap', x: gridX, y: gridY, z: d_blk, colorscale: 'RdBu', zmid:0, zmin:-0.2, zmax:0.2, name:'dBlock', xaxis:'x5', yaxis:'y5'}};
        const trace6 = {{type: 'heatmap', x: gridX, y: gridY, z: d_acc, colorscale: 'RdBu', zmid:0, zmin:-0.2, zmax:0.2, name:'dAcc', xaxis:'x6', yaxis:'y6'}};
        const trace7 = {{type: 'heatmap', x: gridX, y: gridY, z: d_fin, colorscale: 'RdBu', zmid:0, zmin:-0.2, zmax:0.2, name:'dFin', xaxis:'x7', yaxis:'y7'}};
        const trace8 = {{type: 'heatmap', x: gridX, y: gridY, z: d_xg, colorscale: 'RdBu', zmid:0, zmin:-0.1, zmax:0.1, name:'dxG', xaxis:'x8', yaxis:'y8'}};
        
        const layout = {{
             grid: {{rows: 2, columns: 4, pattern: 'independent'}},
             paper_bgcolor: '#111', plot_bgcolor: '#111',
             font: {{color: 'white'}},
             title: 'Nested Model Client-Side',
             // Layout shapes: We need to assign them to all axes
             shapes: []
        }};
        
        // Replicate Shapes for all 8 subplots
        ['','2','3','4','5','6','7','8'].forEach(suffix => {{
            RINK_SHAPES.forEach(s => {{
                let s2 = {{...s}};
                s2.xref = 'x' + suffix;
                s2.yref = 'y' + suffix;
                layout.shapes.push(s2);
            }});
        }});
        
        // Hide axes lines
        for(let i=1; i<=8; i++) {{
            let s = (i===1) ? '' : i;
            layout['xaxis'+s] = {{showgrid: false, zeroline: false, range: [0, 100], visible: false}};
            layout['yaxis'+s] = {{showgrid: false, zeroline: false, range: [-42.5, 42.5], visible: false, scaleanchor:'x'+s}};
        }}
        
        // Annotations for Titles
        layout.annotations = [
            {{text:'Block Prob', x:0.1, y:1.05, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'Accuracy', x:0.37, y:1.05, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'Finish', x:0.63, y:1.05, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'xG', x:0.9, y:1.05, showarrow:false, xref:'paper', yref:'paper'}},
            
            {{text:'Δ Block', x:0.1, y:0.43, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'Δ Accuracy', x:0.37, y:0.43, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'Δ Finish', x:0.63, y:0.43, showarrow:false, xref:'paper', yref:'paper'}},
            {{text:'Δ xG', x:0.9, y:0.43, showarrow:false, xref:'paper', yref:'paper'}},
        ];

        Plotly.react('plot', [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8], layout);
    }}
    
    function convertTo2D(flat) {{
        let out = [];
        for(let r=0; r<Y_POINTS; r++) {{
            let row = [];
            for(let c=0; c<X_POINTS; c++) {{
                row.push(flat[r*X_POINTS + c]);
            }}
            out.push(row);
        }}
        return out;
    }}
    
    function calcDelta(curr, base) {{
        let out = [];
        for(let r=0; r<curr.length; r++) {{
             let row = [];
             for(let c=0; c<curr[0].length; c++) {{
                 row.push(curr[r][c] - base[r][c]);
             }}
             out.push(row);
        }}
        return out;
    }}
    
    function setBaseline() {{
        const inputs = getInputs();
        const res = predict_scenario(inputs);
        baselineData = [
            convertTo2D(res[0]), convertTo2D(res[1]), convertTo2D(res[2]), convertTo2D(res[3])
        ];
        alert("Baseline Set!");
        updatePlot();
    }}
    
    function clearBaseline() {{
        baselineData = null;
        updatePlot();
    }}

    // Start
    init();
</script>
</body>
</html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    main()
