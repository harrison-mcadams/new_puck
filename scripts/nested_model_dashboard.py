
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

def extract_pipeline_params(pipeline, feature_names):
    """Extracts weights, scales, knots, etc from a single layer pipeline."""
    
    # 1. Pipeline Steps
    # Structure: preprocessor (ColumnTransformer) -> clf (LogisticRegression)
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    
    # 2. Extract Coefficients
    coef = clf.coef_[0].tolist()
    intercept = clf.intercept_[0]
    
    # 3. Extract Transformers
    # ColumnTransformer transformers_: [('num', Pipeline, [cols]), ('cat', Pipeline, [cols])]
    
    transformers = {}
    
    # Helper to map input feature names to their output implementations
    feature_map = [] # Ordered list of operations to produce the final feature vector
    
    current_coef_idx = 0
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder': continue
        
        # 'trans' is a Pipeline
        # Numeric pipeline: [imputer, (poly/spline), scaler]
        # Cat pipeline: [imputer, ohe]
        
        # We need to preserve the ORDER because Coefs match the concatenations
        
        if name == 'num':
            # Numeric Pipeline
            # Check for Spline or Poly
            spline = None
            poly = None
            scaler = None
            
            for step_name, step in trans.steps:
                if isinstance(step, SplineTransformer):
                    spline = step
                elif isinstance(step, PolynomialFeatures):
                    poly = step
                elif isinstance(step, StandardScaler):
                    scaler = step
            
            # Extract Params
            if spline:
                # SplineTransformer output size per feature: n_knots + degree - 1
                # (default n_knots=7, degree=3 -> 9 features per input)
                # But include_bias=False? Default True in newer sklearn?
                # Code says: include_bias=False
                knots = spline.bsplines_[0].t.tolist() # Uniform knots usually same for all? 
                # Actually, sklearn SplineTransformer fits knots PER FEATURE if using quantiles.
                # Check strategy. Default 'uniform'.
                # bsplines_ is list of scipy bsps? No.
                # In sklearn < 1.0 it was different.
                # Let's inspect `bsplines_`
                
                # We need: knots for each feature
                # If uniform, min/max matter.
                
                # Sklearn SplineTransformer logic:
                # processing is independent per feature.
                
                num_feats_data = {}
                
                for idx, col in enumerate(cols):
                    # Find knots for this column
                    # bsplines_ is list of size n_features
                    # Each element has .t (knots), .k (degree)
                    bs = spline.bsplines_[idx]
                    
                    # Scaler mean/scale for the OUTPUT features of this column
                    # The scaler is applied AFTER spline expansion.
                    # Output features for this col: (n_knots + degree - 1) - (1 if no bias)
                    n_out = spline.n_features_out_ // len(cols) # Approximation
                    # Actually better to track indices
                    
                    # We need precise mapping.
                    # Since we implement spline in JS, we need:
                    #  - degree
                    #  - knots
                    #  - scaler means/scales for the expanded features
                    
                    # Get slice of scaler
                    scale_slice_mean = scaler.mean_[current_coef_idx : current_coef_idx + n_out].tolist()
                    scale_slice_scale = scaler.scale_[current_coef_idx : current_coef_idx + n_out].tolist()
                    
                    # Get slice of Coefs
                    coef_slice = coef[current_coef_idx : current_coef_idx + n_out]
                    
                    num_feats_data[col] = {
                        'type': 'spline',
                        'knots': bs.t.tolist(),
                        'degree': bs.k,
                        'coefs': coef_slice,
                        'scaler_mean': scale_slice_mean,
                        'scaler_scale': scale_slice_scale,
                        'idx_start': current_coef_idx
                    }
                    
                    current_coef_idx += n_out
                    
                transformers['num'] = num_feats_data

            elif poly:
                # Poly logic
                # Only 1 feature usually? Or multiple?
                # Poly expands interactions too if multiple num features passed together
                # The NestedGLM splits num features into a block?
                # "num_trans" applies to ALL num_features.
                # So if we have [dist, speed], poly(2) -> d, s, d^2, d*s, s^2
                
                # JS implementation of generic PolyFeatures is tricky if we don't know the mapping.
                # Luckily sklearn provides `get_feature_names_out`.
                
                poly_out_names = poly.get_feature_names_out(cols)
                scaler_mean = scaler.mean_
                scaler_scale = scaler.scale_
                
                # We need to map each output term 'd^2 s' to a coef and scaler
                poly_data = []
                for i, name in enumerate(poly_out_names):
                     poly_data.append({
                         'term': name, # e.g. "distance^2 speed"
                         'coef': coef[current_coef_idx + i],
                         'mean': scaler.mean_[current_coef_idx + i],
                         'scale': scaler.scale_[current_coef_idx + i]
                     })
                
                transformers['num_poly'] = poly_data
                current_coef_idx += len(poly_out_names)

        elif name == 'cat':
            # Categorical Pipeline
            # [imputer, ohe]
            ohe = trans.named_steps['ohe']
            
            # OHE categories_ is list of arrays
            # For each input col, we have categories
            
            cat_feats_data = {}
            
            for idx, col in enumerate(cols):
                cats = ohe.categories_[idx].tolist()
                
                # OHE drops? sparse_output=False, handle_unknown='ignore'.
                # If drop='first'? Check config. Usually None for robustness.
                # Code says: drop=None (default)
                
                feat_dict = {}
                for cat_idx, cat_val in enumerate(cats):
                    # Coef for this specific category
                    c = coef[current_coef_idx]
                    feat_dict[str(cat_val)] = c
                    current_coef_idx += 1
                
                # If handle_unknown='ignore', all 0s -> 0 contribution.
                # Effectively base intercept absorbs unknown if bias?
                
                cat_feats_data[col] = {
                    'type': 'ohe',
                    'weights': feat_dict
                }
                
            transformers['cat'] = cat_feats_data
            
    return {
        'intercept': intercept,
        'transformers': transformers
    }

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    output_path = "analysis/nested_xgs/nested_model_dashboard.html"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    
    print("Extracting model parameters...")
    
    # Extract Metadata
    # We need to know which features go where
    # Model stores 'features' list.
    # The pipeline splits them inside using 'cat_features' listing.
    
    # We have to replicate the split logic to know which features are Num vs Cat
    all_features = model.features
    cat_features_list = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state']
    
    # Refine cat list based on actual features present
    actual_cats = [f for f in cat_features_list if f in all_features]
    actual_nums = [f for f in all_features if f not in actual_cats]
    
    # Also interact_col logic
    if hasattr(model, 'interact_col') and model.interact_col and model.interact_col in all_features:
        # It's numeric
        pass

    # Layers
    layers_data = {
        'block': extract_pipeline_params(model.model_block, [f for f in all_features if f != 'shot_type']),
        'accuracy': extract_pipeline_params(model.model_acc, all_features),
        'finish': extract_pipeline_params(model.model_finish, all_features)
    }
    
    # Priors
    priors = load_data_and_priors(all_features)
    
    # Additional Metadata to export
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
            'rebound_angle_change': 0,
            'rebound_time_diff': 0,
            'last_event_type': 'faceoff' # Lowercase default
        },
        'options': {
            'shooter_role': ['F', 'D'],
            'game_state': ['5v5', '5v4', '4v5', '5v3', '3v5', '4v4', '3v3', '6v5', '5v6', '6v6', 'Marginalized'],
            'shot_type': ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected', 'wrap-around', 'Marginalized'],
            'shoots_catches': ['L', 'R', 'Marginalized'],
            'is_rush': ['0', '1', 'Marginalized'],
            'is_rebound': ['0', '1', 'Marginalized'],
            'last_event_type': sorted(list(priors.get('last_event_type', {}).keys())) + ['Marginalized']
        },
         # Interaction logic flag
        'use_splines': getattr(model, 'use_splines', False),
        'interact_col': getattr(model, 'interact_col', 'dist_angle')
    }
    
    # Include numeric feature default ranges for sliders?
    # Or just use arbitrary inputs in UI?
    # User wanted "All options". For numeric, we can give a slider or input.
    # We will provide default bins [Low, Med, High] for quick select + Slider custom.

    json_str = json.dumps(export_data)
    
    print("Generating HTML...")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Nested Model Dashboard (Client-Side)</title>
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
        <div id="inputs"></div>
        <button class="btn" style="background: #28a745" onclick="setBaseline()">Set Baseline</button>
        <button class="btn" style="background: #dc3545" onclick="clearBaseline()">Clear Top</button>
    </div>
    
    <div id="plot"></div>

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
    
    // Baseline Storage
    let baselineData = null;

    // --- MATH ENGINE ---
    
    function dot(v1, v2) {{
        let s = 0;
        for(let i=0; i<v1.length; i++) s += v1[i] * v2[i];
        return s;
    }}
    
    function sigmoid(z) {{
        return 1 / (1 + Math.exp(-z));
    }}
    
    // BSpline Evaluation (De Boor's Algo simplified for single point or recurring)
    // We need to evaluate basis functions for a value 'x' given knots 't' and degree 'k'
    // Returns array of basis values.
    function bspline_basis(x, t, k) {{
        // t is array of knots. length = n_bases + k + 1
        // output size = len(t) - k - 1
        const n = t.length - k - 1;
        let N = new Array(n).fill(0);
        
        // Find span index i such that t[i] <= x < t[i+1]
        // (with special handling for max value)
        let idx = -1;
        if (x >= t[t.length - k - 1]) {{
             idx = t.length - k - 2; // Last span
        }} else {{
            for(let i=0; i < t.length - 1; i++) {{
                if (x >= t[i] && x < t[i+1]) {{
                    idx = i;
                    break;
                }}
            }}
        }}

        if (idx === -1) return N; // Out of bounds?
        
        // Initialize degree 0
        let b = new Array(k+1).fill(0);
        b[k] = 1; // Corresponding to the span idx
        
        // We only care about basis functions non-zero in this span.
        // There are at most k+1 such functions: N_{{idx-k, k}} ... N_{{idx, k}}
        // Actually sklearn implementation might differ slightly.
        // Let's use generic recursive eval for N_{{i,p}}(x)
        
        // Faster approach:
        // We calculate all non-zero basis functions at x.
        // Returns array of size n, mostly zeros.
        
        // For p=0 to k
        //   Calculate N_{{i,p}}
        
        // Implementation of Cox-De Boor
        // Let's rely on the fact that sklearn splines are standard B-splines.
        
        // Create full N array
        for(let i=0; i<n; i++) {{
           // Determine N_i,k(x)
           // This is slow O(n*k^2).
           // Optimization: Only compute relevant ones.
           N[i] = bspline_recur(i, k, t, x);
        }}
        return N;
    }}
    
    function bspline_recur(i, p, t, x) {{
        if (p === 0) {{
            // N_{{i,0}}(x) = 1 if t[i] <= x < t[i+1], else 0
            // Handle right boundary (== last knot) for last interval
            if (t[i+1] === t[t.length-1] && i === t.length - p - 2) {{
                 return (x >= t[i] && x <= t[i+1]) ? 1 : 0;
            }}
            return (x >= t[i] && x < t[i+1]) ? 1 : 0;
        }} else {{
            let left = 0, right = 0;
            const d1 = t[i+p] - t[i];
            const d2 = t[i+p+1] - t[i+1];
            
            if (d1 > 0) left = ((x - t[i]) / d1) * bspline_recur(i, p-1, t, x);
            if (d2 > 0) right = ((t[i+p+1] - x) / d2) * bspline_recur(i+1, p-1, t, x);
            
            return left + right;
        }}
    }}
    
    function transform_numeric_feature(val, config) {{
        if (config.type === 'spline') {{
            // BSpline Expansion
            // 1. Basis
            let basis = bspline_basis(val, config.knots, config.degree);
            
            // 2. Align dimensions
            // Python SplineTransformer(include_bias=False) seems to keep the first basis function
            // and drop the last one (or similar), based on empirical verification.
            // We must match the scaler_mean length.
            if (basis.length > config.scaler_mean.length) {{
                // Keep the first N elements
                basis = basis.slice(0, config.scaler_mean.length);
            }}
            
            // 3. Scale
            let out = [];
            for(let i=0; i<basis.length; i++) {{
                out.push( (basis[i] - config.scaler_mean[i]) / config.scaler_scale[i] );
            }}
            return out; // Vector
        }}
        return [0];
    }}
    
    function get_layer_score(features_dict, layer_name) {{
        const layer = MODEL.layers[layer_name];
        if (!layer) return -999;
        
        let score = layer.intercept;
        
        // Numeric
        const num_trans = layer.transformers.num;
        for (const [feat_name, config] of Object.entries(num_trans)) {{
            let val = features_dict[feat_name];
            
            // Calculate vector
            let vec = transform_numeric_feature(val, config);
            
            // Dot with coefs
            // Coefs are slice stored in config
            for(let i=0; i<vec.length; i++) {{
                score += vec[i] * config.coefs[i];
            }}
        }}

        // Categorical
        const cat_trans = layer.transformers.cat;
        for (const [feat_name, config] of Object.entries(cat_trans)) {{
            let val = String(features_dict[feat_name]);
            let w = config.weights[val];
            if (w !== undefined) {{
                score += w;
            }}
            // else 0 (unknown/reference)
        }}
        
        return score;
    }}
    
    function predict_scenario(inputs) {{
        // Returns [p_block, p_acc, p_finish, p_xg]
        
        // ... (standard inputs logic)
        
        // Recursive Marginalization Setup
        let marg_keys = [];
        let fixed_inputs = {{...inputs}};
        
        for (const k in inputs) {{
            if (inputs[k] === 'Marginalized' && MODEL.options[k]) {{
                marg_keys.push(k);
            }}
        }}
        
        // Resolve Non-Marginalized Inputs first
        for (const k in fixed_inputs) {{
            if (marg_keys.includes(k)) continue;
            
            // Check if it's a numeric keyword
            if (['Low', 'Med', 'High', 'Marginalized'].includes(fixed_inputs[k])) {{
                fixed_inputs[k] = 0.0; // Todo: embed stats
            }}
        }}
        
        // Prepare Marginalization Loop
        let scenarios = [{{weight: 1.0, inputs: fixed_inputs}}];
        
        for (const key of marg_keys) {{
            let new_scenarios = [];
            let opts = Object.keys(MODEL.priors[key] || {{}});
            if (opts.length === 0) opts = MODEL.options[key].filter(x => x!=='Marginalized');
            
            for (const scen of scenarios) {{
                for (const opt of opts) {{
                    let p = (MODEL.priors[key] && MODEL.priors[key][opt]) || (1.0/opts.length);
                    let s = {{
                        weight: scen.weight * p,
                        inputs: {{...scen.inputs}}
                    }};
                    s.inputs[key] = opt;
                    new_scenarios.push(s);
                }}
            }}
            scenarios = new_scenarios;
        }}
        
        // Prune low weight scenarios
        scenarios = scenarios.filter(s => s.weight > 0.001);
        
        // Now we calculate Grids for each scenario and sum
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
            
            // Calc Grid for this atomic scenario
            // X, Y loops
            for(let r=0; r<H; r++) {{
                for(let c=0; c<W; c++) {{
                    const x = gridX[c];
                    const y = gridY[r];
                    const idx = r*W + c;
                    
                    // Geometry
                    // Geometry
                    const dist = Math.sqrt((x-89)**2 + y**2);
                    
                    // Angle match Python (puck/data_pipeline.py)
                    // Reference vector (0, -1) -> South
                    const dx = x - 89;
                    const dy = y;
                    const angle_rad = Math.atan2(dx, -dy);
                    let angle_deg = -angle_rad * 180 / Math.PI;
                    angle_deg = angle_deg % 360;
                    if (angle_deg < 0) angle_deg += 360;
                    
                    // Apply Interaction
                    let feats = {{...scen.inputs}};
                    feats.distance = dist;
                    feats.angle_deg = angle_deg;
                    feats.dist_angle = dist * Math.abs(angle_deg);
                    
                    // Layers
                    let s_block = get_layer_score(feats, 'block');
                    let p_block = sigmoid(s_block);
                    
                    let s_acc = get_layer_score(feats, 'accuracy');
                    let p_acc = sigmoid(s_acc);
                    
                    let s_fin = get_layer_score(feats, 'finish');
                    let p_fin = sigmoid(s_fin);
                    
                    let p_xg = (1 - p_block) * p_acc * p_fin;
                    
                    Z_block[idx] += p_block * w;
                    Z_acc[idx] += p_acc * w;
                    Z_fin[idx] += p_fin * w;
                    Z_xg[idx] += p_xg * w;
                }}
            }}
        }}
        
        // Normalize
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
            'Context': ['game_state', 'score_diff', 'period_number'],
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
