"""non_nested_model_dashboard.py

Generates an interactive HTML dashboard for the NonNestedGLM tensor-product xG model.
Directly parallel to nested_model_dashboard.py, but for a single-layer model.
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

from puck import data_pipeline, fit_glm, fit_xgs, features as feature_util, rink

import math
try:
    from puck.rink import calculate_distance_and_angle
except ImportError:
    def calculate_distance_and_angle(x, y, goal_x, goal_y=0.0):
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
    """Computes the spatial component score for the standard 50x43 grid."""
    X_POINTS = 50
    Y_POINTS = 43
    xs = np.linspace(0, 100, X_POINTS)
    ys = np.linspace(-42.5, 42.5, Y_POINTS)
    
    clf = pipeline.named_steps['clf']
    pre = pipeline.named_steps['preprocessor']
    
    try:
        tensor_transformer = None
        for name, trans, cols in pre.transformers_:
            if name == 'spatial_tensor':
                tensor_transformer = trans
                break
        
        if tensor_transformer is None:
            return None
            
        all_out_feats = pre.get_feature_names_out()
        spatial_indices = [i for i, f in enumerate(all_out_feats) if f.startswith('spatial_tensor__')]
        
        if not spatial_indices:
            return None

        spatial_coefs = clf.coef_[0][spatial_indices]
        
        grid_rows = []
        for y in ys:
            for x in xs:
                dist, ang = calculate_distance_and_angle(x, y, 89.0, 0.0)
                grid_rows.append({'distance': dist, 'angle_deg': ang})
        
        grid_df = pd.DataFrame(grid_rows)
        X_trans = tensor_transformer.transform(grid_df)
        scores = X_trans @ spatial_coefs
        
        p1, p99 = np.percentile(scores, [1, 99])
        scores = np.clip(scores, p1, p99)
        
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
        df = fit_xgs.load_data()
        priors = {}
        for f in features:
            if f in df.columns:
                if df[f].dtype == object or df[f].nunique() < 20: 
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
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    coef = clf.coef_[0].tolist()
    intercept = clf.intercept_[0]
    transformers = {}
    current_coef_idx = 0
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder': continue
        n_out = 0
        if hasattr(trans, 'get_feature_names_out'):
             try:
                 out_names = trans.get_feature_names_out()
             except:
                 try:
                    out_names = trans.get_feature_names_out(cols)
                 except: 
                     out_names = []
             n_out = len(out_names)
        elif hasattr(trans, 'categories_'): # OHE direct
             n_out = sum(len(c) for c in trans.categories_)
        
        if name == 'spatial_tensor':
             current_coef_idx += n_out
             continue
             
        block_coefs = coef[current_coef_idx : current_coef_idx + n_out]
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
             scaler = trans.named_steps['scaler']
             feats_data = {}
             if len(cols) > 0:
                 n_per_col = n_out // len(cols)
                 local_idx = 0
                 for i, col in enumerate(cols):
                     col_c = block_coefs[local_idx : local_idx + n_per_col]
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
        current_coef_idx += n_out

    spatial_grid = compute_spatial_grid(pipeline, features)
    return {
        'intercept': intercept,
        'transformers': transformers,
        'spatial_grid': spatial_grid
    }

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "analysis/xgs/xg_model_non_nested_tensor.joblib"
        
    base_name = os.path.basename(model_path).replace('.joblib', '')
    output_path = f"analysis/non_nested_xgs/{base_name}_dashboard.html"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    print("Extracting model parameters...")
    
    all_features = model.features
    cat_features_list = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state', 'relative_game_state']
    actual_cats = [f for f in cat_features_list if f in all_features]
    actual_nums = [f for f in all_features if f not in actual_cats]

    # Non-Nested has only one model layer
    layers_data = {
        'xg': extract_pipeline_params(model.model, all_features)
    }
    
    priors = load_data_and_priors(all_features)
    
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
            'last_event_type': 'faceoff',
            'relative_game_state': '5v5' 
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
            'last_event_type': sorted(list(priors.get('last_event_type', {}).keys())) + ['Marginalized'],
            'relative_game_state': ['5v5', '5v4', '4v5', '5v3', '3v5', '4v4', '3v3', '6v5', '5v6', '6v6', 'Marginalized']
        },
        'use_splines': getattr(model, 'use_splines', False)
    }
    
    json_str = json.dumps(export_data)
    print(f"Generating HTML to {output_path}...")
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Non-Nested Model Dashboard - {{BASE_NAME}}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { margin: 0; background: #111; color: white; font-family: sans-serif; overflow: hidden; }
        #controls { 
            position: absolute; top: 0; left: 0; width: 300px; height: 100vh; 
            background: #222; overflow-y: auto; padding: 10px; box-sizing: border-box; 
            border-right: 1px solid #444;
        }
        #plot { 
            position: absolute; top: 0; left: 300px; right: 0; bottom: 0; 
        }
        .ctrl-group { margin-bottom: 12px; }
        label { display: block; font-size: 0.85em; color: #aaa; margin-bottom: 3px; }
        select, input { 
            width: 100%; background: #333; color: white; border: 1px solid #555; 
            padding: 4px; border-radius: 3px; 
        }
        .btn { 
            width: 48%; padding: 8px; border: none; cursor: pointer; color: white; margin-top: 10px;
        }
        #loading {
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            font-size: 2em; background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px;
            z-index: 1000; display: block;
        }
    </style>
</head>
<body>
    <div id="loading">Loading Engine...</div>
    
    <div id="controls">
        <h3>Model Controls</h3>
        <small>{{BASE_NAME}} (Non-Nested)</small>
        <div id="inputs"></div>
        <button class="btn" style="background: #28a745" onclick="setBaseline()">Set Baseline</button>
        <button class="btn" style="background: #dc3545" onclick="clearBaseline()">Clear Top</button>
    </div>
    
    <div id="plot"></div>

<script>
    const MODEL = {{JSON_STR}};
    const RINK_SHAPES = {{RINK_SHAPES}};
    const X_POINTS = 50;
    const Y_POINTS = 43;
    let gridX = [];
    let gridY = [];
    for(let i=0; i<X_POINTS; i++) gridX.push(i * (100/(X_POINTS-1)));
    for(let i=0; i<Y_POINTS; i++) gridY.push(-42.5 + i * (85/(Y_POINTS-1)));
    let baselineData = null;

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
    
    function bspline_basis(x, t, k) {
        const n = t.length - k - 1;
        let N = new Array(n).fill(0);
        let idx = -1;
        if (x < t[0] || x > t[t.length-1]) return N;
        if (x === t[t.length - 1]) { idx = t.length - k - 2; } 
        else {
            for(let i=0; i < t.length - 1; i++) {
                if (x >= t[i] && x < t[i+1]) { idx = i; break; }
            }
        }
        if (idx === -1) return N;
        let basis = new Array(k + 1).fill(0);
        let left = new Array(k + 1).fill(0);
        let right = new Array(k + 1).fill(0);
        basis[0] = 1.0;
        for (let j = 1; j <= k; j++) {
            left[j] = x - t[idx + 1 - j];
            right[j] = t[idx + j] - x;
            let saved = 0.0;
            for (let r = 0; r < j; r++) {
                let term = basis[r] / (right[r + 1] + left[j - r]);
                basis[r] = saved + right[r + 1] * term;
                saved = left[j - r] * term;
            }
            basis[j] = saved;
        }
        for(let j = 0; j <= k; j++) {
            let map_idx = idx - k + j;
            if (map_idx >= 0 && map_idx < n) { N[map_idx] = basis[j]; }
        }
        return N;
    }
    
    function transform_numeric_feature(val, config) {
        if (config.type === 'spline') {
            let basis = bspline_basis(val, config.knots, config.degree);
            if (basis.length > config.scaler_mean.length) { basis = basis.slice(0, config.scaler_mean.length); }
            let out = [];
            for(let i=0; i<basis.length; i++) {
                out.push( (basis[i] - config.scaler_mean[i]) / config.scaler_scale[i] );
            }
            return out;
        }
        return [0];
    }
    
    function get_layer_score(features_dict, layer_name, grid_r, grid_c) {
        const layer = MODEL.layers[layer_name];
        if (!layer) return -999;
        let score = layer.intercept;
        const trans = layer.transformers;
        if (layer.spatial_grid && grid_r !== undefined) {
             score += layer.spatial_grid[grid_r][grid_c];
        }
        if (trans.num_spline) {
             for (const [feat_name, config] of Object.entries(trans.num_spline)) {
                let vec = transform_numeric_feature(features_dict[feat_name], config);
                for(let i=0; i<vec.length; i++) { score += vec[i] * config.coefs[i]; }
             }
        }
        if (trans.cat) {
            for (const [feat_name, config] of Object.entries(trans.cat)) {
                let w = config.weights[String(features_dict[feat_name])];
                if (w !== undefined) score += w;
            }
        }
        if (trans.binary) {
            for (const [feat_name, config] of Object.entries(trans.binary)) {
                let val = parseFloat(features_dict[feat_name]) || 0;
                score += ((val - config.scaler_mean) / config.scaler_scale) * config.coef;
            }
        }
        return score;
    }
    
    function predict_scenario(inputs) {
        let marg_keys = [];
        let fixed_inputs = {...inputs};
        for (const k in inputs) {
            if (inputs[k] === 'Marginalized' && MODEL.options[k]) { marg_keys.push(k); }
        }
        let scenarios = [{weight: 1.0, inputs: fixed_inputs}];
        for (const key of marg_keys) {
            let new_scenarios = [];
            let opts = Object.keys(MODEL.priors[key] || {});
            if (opts.length === 0) opts = MODEL.options[key].filter(x => x!=='Marginalized');
            for (const scen of scenarios) {
                for (const opt of opts) {
                    let p = (MODEL.priors[key] && MODEL.priors[key][opt]) || (1.0/opts.length);
                    let s = { weight: scen.weight * p, inputs: {...scen.inputs} };
                    s.inputs[key] = opt;
                    new_scenarios.push(s);
                }
            }
            scenarios = new_scenarios;
        }
        scenarios = scenarios.filter(s => s.weight > 0.001);
        let H = Y_POINTS, W = X_POINTS;
        let Z_xg = new Float32Array(H*W);
        let total_weight = 0;
        for (const scen of scenarios) {
            const w = scen.weight;
            total_weight += w;
            for(let r=0; r<H; r++) {
                for(let c=0; c<W; c++) {
                    const x = gridX[c], y = gridY[r];
                    let feats = {...scen.inputs};
                    feats.distance = Math.sqrt((x-89)**2 + y**2);
                    feats.angle_deg = (-(Math.atan2(x-89, -y) * 180 / Math.PI)) % 360;
                    if (feats.angle_deg < 0) feats.angle_deg += 360;
                    Z_xg[r*W + c] += sigmoid(get_layer_score(feats, 'xg', r, c)) * w;
                }
            }
        }
        if(total_weight > 0) { for(let i=0; i<Z_xg.length; i++) Z_xg[i] /= total_weight; }
        return Z_xg;
    }

    function init() {
        const inputDiv = document.getElementById('inputs');
        const groups = {
            'Context': ['game_state', 'relative_game_state', 'score_diff', 'period_number', 'is_home'],
            'Shooter': ['shooter_role', 'shoots_catches', 'shot_type'],
            'Play Info': ['is_rush', 'is_rebound', 'last_event_type', 'speed_from_last_event']
        };
        for(const [gname, fields] of Object.entries(groups)) {
            let fs = document.createElement('fieldset');
            fs.className = 'ctrl-group';
            fs.innerHTML = `<legend>${gname}</legend>`;
            fields.forEach(f => {
                if (!MODEL.features.includes(f) && f !== 'score_diff') return;
                let wrap = document.createElement('div');
                let lbl = document.createElement('label'); lbl.innerText = f;
                let sel = document.createElement('select'); sel.id = 'in_' + f; sel.onchange = updatePlot;
                let opts = MODEL.options[f] || ['Marginalized', 'Low', 'Med', 'High'];
                opts.forEach(o => {
                    let opt = document.createElement('option'); opt.value = o; opt.innerText = o;
                    sel.appendChild(opt);
                });
                if (MODEL.defaults[f]) sel.value = MODEL.defaults[f];
                wrap.appendChild(lbl); wrap.appendChild(sel); fs.appendChild(wrap);
            });
            inputDiv.appendChild(fs);
        }
        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }
    
    function getInputs() {
        let inp = {};
        document.querySelectorAll('select').forEach(s => { inp[s.id.substring(3)] = s.value; });
        for (const k in MODEL.defaults) { if (inp[k] === undefined) inp[k] = MODEL.defaults[k]; }
        return inp;
    }
    
    function updatePlot() {
        const inputs = getInputs();
        const z_xg_flat = predict_scenario(inputs);
        const z_xg = convertTo2D(z_xg_flat);
        let d_xg = baselineData ? calcDelta(z_xg, baselineData) : z_xg.map(r => r.map(c => 0));
        
        const trace1 = {type: 'heatmap', x: gridX, y: gridY, z: z_xg, colorscale: 'Plasma', zmin:0, zmax:0.3, name:'xG', xaxis:'x', yaxis:'y'};
        const trace2 = {type: 'heatmap', x: gridX, y: gridY, z: d_xg, colorscale: 'RdBu', zmid:0, zmin:-0.1, zmax:0.1, name:'dxG', xaxis:'x2', yaxis:'y2'};
        
        const layout = {
             grid: {rows: 1, columns: 2, pattern: 'independent'},
             paper_bgcolor: '#111', plot_bgcolor: '#111', font: {color: 'white'},
             title: 'Non-Nested GLM xG Dashboard',
             shapes: []
        };
        
        ['','2'].forEach(suffix => {
            RINK_SHAPES.forEach(s => {
                let s2 = {...s}; s2.xref = 'x' + suffix; s2.yref = 'y' + suffix;
                layout.shapes.push(s2);
            });
            layout['xaxis'+suffix] = {range: [0, 100], visible: false};
            layout['yaxis'+suffix] = {range: [-42.5, 42.5], visible: false, scaleanchor:'x'+suffix};
        });
        
        layout.annotations = [
            {text:'xG Probability', x:0.25, y:1.05, showarrow:false, xref:'paper', yref:'paper'},
            {text:'Δ xG Relative to Baseline', x:0.75, y:1.05, showarrow:false, xref:'paper', yref:'paper'},
        ];
        Plotly.react('plot', [trace1, trace2], layout);
    }
    
    function convertTo2D(flat) {
        let out = [];
        for(let r=0; r<Y_POINTS; r++) {
            let row = [];
            for(let c=0; c<X_POINTS; c++) row.push(flat[r*X_POINTS + c]);
            out.push(row);
        }
        return out;
    }
    
    function calcDelta(curr, base) {
        return curr.map((r, i) => r.map((c, j) => c - base[i][j]));
    }
    
    function setBaseline() {
        baselineData = convertTo2D(predict_scenario(getInputs()));
        alert("Baseline Set!");
        updatePlot();
    }
    
    function clearBaseline() { baselineData = null; updatePlot(); }

    init();
</script>
</body>
</html>
    """
    
    html_content = html_template.replace('{{BASE_NAME}}', base_name).replace('{{JSON_STR}}', json_str).replace('{{RINK_SHAPES}}', json.dumps(get_rink_shapes()))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    main()

