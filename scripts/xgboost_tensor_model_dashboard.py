"""xgboost_tensor_model_dashboard.py

Generates an interactive HTML dashboard for the XGBoost Alternate xG model.
Functional Parity: Uses a client-side JavaScript tree inference engine.
No GLM Baselines: Relies on pure XGBoost spatial features.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgboost_tensor, config as puck_config, data_pipeline

def json_serializable(obj):
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_booster_data(model, feature_names=None):
    """Extracts JSON dump and feature names from an XGBClassifier."""
    if model is None:
        return None
    booster = model.get_booster()
    if feature_names:
        booster.feature_names = feature_names
    trees_json = booster.get_dump(dump_format='json')
    trees = [json.loads(t) for t in trees_json]
    
    try:
        config = json.loads(booster.save_config())
        base_score_str = config['learner']['learner_model_param']['base_score']
        if base_score_str.startswith('[') and base_score_str.endswith(']'):
            base_score = float(base_score_str[1:-1])
        else:
            base_score = float(base_score_str)
    except Exception:
        base_score = 0.5
        
    return {
        'trees': trees,
        'feature_names': booster.feature_names or [],
        'base_score': base_score
    }

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

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor.joblib')
    base_name = os.path.basename(model_path).replace('.joblib', '')
    output_path = f"analysis/xgboost_tensor_xgs/{base_name}_dashboard.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    # Gather Data
    export_data = {
        'model_name': base_name,
        'features': model.features,
        'vocabs': fit_xgboost_tensor.CATEGORICAL_VOCABS,
        'priors': model.categorical_priors_,
        'layers': {
            'block': extract_booster_data(model.model_block, getattr(model, 'features_block', model.features)),
            'accuracy': extract_booster_data(model.model_acc, getattr(model, 'features_acc', model.features)),
            'finish': extract_booster_data(model.model_finish, getattr(model, 'features_fin', model.features))
        },
        'spline': {
            'use': getattr(model, 'use_splines', False),
            'feature_names': getattr(model, 'spline_feature_names_', [])
        },
        'calibrators': {},
        'defaults': {
            'distance': 25.0, 'angle_deg': 0.0, 'game_state': '5v5', 'relative_game_state': '5v5',
            'shot_type': 'wrist', 'shooter_role': 'F', 'shoots_catches': 'L',
            'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
            'period_number': 2, 'speed_from_last_event': 7.5, 'last_event_type': 'giveaway',
            'dist_from_last_event': 15.0, 'last_event_time_diff': 2.0
        },
        'numeric_defaults': data_pipeline.NUMERIC_DEFAULTS,
        'options': {str(k): list(v) + ['Marginalized'] for k, v in fit_xgboost_tensor.CATEGORICAL_VOCABS.items()},
        'presets': {
            'Owen Tippett (Clean Shot)': {
                'x': 78, 'y': 10, 'shot_type': 'wrist', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 0, 'period_number': 2, 'score_diff': 0,
                'last_event_type': 'giveaway', 'last_event_time_diff': 2.0, 'dist_from_last_event': 30.0, 'speed_from_last_event': 15.0
            },
            'Classic Point Shot': {
                'x': 28, 'y': 25, 'shot_type': 'slap', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 0, 'period_number': 1, 'score_diff': 0,
                'last_event_type': 'faceoff', 'last_event_time_diff': 1.5, 'dist_from_last_event': 40.0, 'speed_from_last_event': 25.0
            },
            'High-Danger Rush': {
                'x': 75, 'y': -5, 'shot_type': 'snap', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 1, 'is_rebound': 0, 'period_number': 3, 'score_diff': -1,
                'last_event_type': 'takeaway', 'last_event_time_diff': 3.0, 'dist_from_last_event': 60.0, 'speed_from_last_event': 35.0
            },
            'Rebound Scramble': {
                'x': 85, 'y': 2, 'shot_type': 'backhand', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 1, 'period_number': 2, 'score_diff': 1,
                'last_event_type': 'shot-on-goal', 'last_event_time_diff': 0.8, 'dist_from_last_event': 5.0, 'speed_from_last_event': 5.0
            }
        }
    }
    
    # Add numerical options
    extra_options = {
        'is_rush': [0, 1, 'Marginalized'],
        'is_rebound': [0, 1, 'Marginalized'],
        'is_home': [0, 1, 'Marginalized'],
        'period_number': [1, 2, 3, 4],
        'score_diff': [-3, -2, -1, 0, 1, 2, 3]
    }
    for k, v in extra_options.items():
        export_data['options'][k] = v

    # --- Pre-calculate Basis Grid ---
    if export_data['spline']['use']:
        print("Calculating Spline Basis Lookup Table...")
        # Mirror grid from JS: X_POINTS=50, Y_POINTS=43
        grid_x = np.linspace(0, 100, 50)
        grid_y = np.linspace(-42.5, 42.5, 43)
        
        # Flattened grid for transformer
        xx, yy = np.meshgrid(grid_x, grid_y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        basis = model.spline_transformer_.transform(pd.DataFrame(points, columns=['x', 'y']))
        # basis shape: (50*43, 49)
        export_data['spline']['basis_lookup'] = basis.tolist() # [pixel_idx][basis_idx]

    json_data = json.dumps(json_serializable(export_data))
    rink_shapes_json = json.dumps(get_rink_shapes())

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>XGBoost Alternate Dashboard | __MODEL_NAME__</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { margin: 0; background: #111; color: white; font-family: 'Inter', system-ui, -apple-system, sans-serif; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #controls { width: 320px; background: #1a1a1a; padding: 20px; overflow-y: auto; border-right: 1px solid #333; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index: 10; }
        #plot-area { flex: 1; position: relative; background: #111; }
        #plot { width: 100%; height: 100%; }
        .ctrl-group { margin-bottom: 20px; padding: 15px; border: 1px solid #333; border-radius: 8px; background: #222; }
        .ctrl-group legend { padding: 0 10px; font-weight: bold; color: #aaa; font-size: 0.9em; text-transform: uppercase; }
        .field { margin-bottom: 12px; }
        label { display: block; font-size: 0.8em; color: #888; margin-bottom: 4px; }
        select { width: 100%; background: #333; color: white; border: 1px solid #444; padding: 6px; border-radius: 4px; box-sizing: border-box; }
        input[type=range] { width: 100%; margin-top: 8px; -webkit-appearance: none; background: #444; height: 4px; border-radius: 2px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; background: #00ff88; border-radius: 50%; cursor: pointer; }
        .field label span { font-weight: bold; color: #00ff88; float: right; }
        .btn-row { display: flex; gap: 10px; margin-top: 20px; }
        button { flex: 1; padding: 10px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; transition: opacity 0.2s; }
        .btn-baseline { background: #2d5a27; color: #fff; }
        .btn-clear { background: #5a2727; color: #fff; }
        #loading { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; justify-content: center; align-items: center; z-index: 1000; font-size: 1.5em; }
    </style>
</head>
<body>
    <div id="loading">Initializing Engine...</div>
    <div id="container">
        <div id="controls">
            <h2 style="margin-top:0; color: #00ff88; font-size: 1.2em;">XGBoost Alternate (Pure Spatial)</h2>
            <div id="inputs-container"></div>
            <div class="btn-row">
                <button class="btn-baseline" onclick="setBaseline()">Set Baseline</button>
                <button class="btn-clear" onclick="clearBaseline()">Clear Δ</button>
            </div>
            <div style="margin-top: 20px; font-size: 0.7em; color: #555;">
                Engine: Client-Side Tree Traversal (Pure Spatial)
            </div>
        </div>
        <div id="plot-area">
            <div id="plot"></div>
        </div>
    </div>

<script>
    const MODEL = __JSON_DATA__;
    const RINK_SHAPES = __RINK_SHAPES__;
    const X_POINTS = 50, Y_POINTS = 43;
    
    let gridX = [], gridY = [];
    for(let i=0; i<X_POINTS; i++) gridX.push(i * (100/(X_POINTS-1)));
    for(let i=0; i<Y_POINTS; i++) gridY.push(-42.5 + i * (85/(Y_POINTS-1)));
    
    let baselineData = null;

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function evaluateTree(node, featureValues) {
        if (!node) return 0;
        if (node.leaf !== undefined) return node.leaf;
        
        const fName = node.split;
        const val = featureValues[fName];
        
        if (val === null || val === undefined || isNaN(val)) {
            const defId = node.default;
            const child = node.children.find(c => String(c.nodeid) == String(defId));
            return evaluateTree(child, featureValues);
        }
        
        if (Array.isArray(node.split_condition)) {
            if (node.split_condition.includes(val)) {
                const child = node.children.find(c => String(c.nodeid) == String(node.yes));
                return evaluateTree(child, featureValues);
            } else {
                const child = node.children.find(c => String(c.nodeid) == String(node.no));
                return evaluateTree(child, featureValues);
            }
        }
        
        if (node.split_type === 'categorical') {
            const cats = node.split_categories || [];
            if (cats.includes(val)) {
                const child = node.children.find(c => String(c.nodeid) == String(node.yes));
                return evaluateTree(child, featureValues);
            } else {
                const child = node.children.find(c => String(c.nodeid) == String(node.no));
                return evaluateTree(child, featureValues);
            }
        } else {
            if (val < node.split_condition) {
                const child = node.children.find(c => String(c.nodeid) == String(node.yes));
                return evaluateTree(child, featureValues);
            } else {
                const child = node.children.find(c => String(c.nodeid) == String(node.no));
                return evaluateTree(child, featureValues);
            }
        }
    }

    function evaluateForest(layerName, featureValues) {
        const forest = MODEL.layers[layerName];
        if (!forest) return 0;
        let margin = Math.log(forest.base_score / (1 - forest.base_score));
        for (const tree of forest.trees) {
            margin += evaluateTree(tree, featureValues);
        }
        return margin;
    }

    function predictScenario(inputs) {
        try {
            let baseFeatures = {...inputs};
            for (const fName in MODEL.vocabs) {
                const val = baseFeatures[fName];
                if (val === 'Marginalized') baseFeatures[fName] = null;
                else if (typeof val === 'string') {
                    const v_idx = MODEL.vocabs[fName].indexOf(val);
                    baseFeatures[fName] = (v_idx === -1) ? null : v_idx;
                }
            }
            MODEL.features.forEach(f => {
                if (baseFeatures[f] === undefined) {
                    baseFeatures[f] = MODEL.numeric_defaults[f] !== undefined ? MODEL.numeric_defaults[f] : 0.0;
                }
                
                // --- HARDENED NUMERIC CASTING ---
                // Features like is_rush, is_rebound, is_home, score_diff, period_number 
                // come from <select> as strings. We must ensure they are numeric for the trees.
                if (baseFeatures[f] !== null && baseFeatures[f] !== undefined && baseFeatures[f] !== 'Marginalized') {
                    if (!MODEL.vocabs[f]) {
                        const num = Number(baseFeatures[f]);
                        if (!isNaN(num)) baseFeatures[f] = num;
                    }
                } else if (baseFeatures[f] === 'Marginalized') {
                    baseFeatures[f] = null;
                }
            });

            // Debug first pixel
            console.log("Scenario Inputs:", inputs);
            console.log("Processed Base Features:", baseFeatures);

            let H = Y_POINTS, W = X_POINTS;
            let Z_block = new Float32Array(H*W), Z_acc = new Float32Array(H*W), Z_fin = new Float32Array(H*W), Z_xg = new Float32Array(H*W);
            
            for(let r=0; r<H; r++) {
                for(let c=0; c<W; c++) {
                    const idx = r*W + c;
                    let features = {...baseFeatures};
                    const x = gridX[c], y = gridY[r];
                    const x_safe = Math.max(0, Math.min(x, 100));
                    const y_safe = Math.max(-42.5, Math.min(y, 42.5));
                    const dist = Math.sqrt((x_safe - 89)**2 + y_safe**2);
                    const angle_rad = Math.atan2(x_safe - 89, -y_safe);
                    let angle_deg = ((-angle_rad * 180 / Math.PI) % 360 + 360) % 360;
                    
                    features.distance = dist;
                    features.angle_deg = angle_deg;
                    // Sync raw x,y in case model uses them directly
                    if (features.x !== undefined) features.x = x_safe;
                    if (features.y !== undefined) features.y = y_safe;
                    
                    if (MODEL.spline.use && MODEL.spline.basis_lookup) {
                        const basis = MODEL.spline.basis_lookup[idx];
                        MODEL.spline.feature_names.forEach((name, i) => {
                            features[name] = basis[i];
                        });
                    }

                    const m_block = evaluateForest('block', features);
                    const m_acc = evaluateForest('accuracy', features);
                    const m_fin = evaluateForest('finish', features);
                    
                    const p_block = sigmoid(m_block);
                    const p_acc = sigmoid(m_acc);
                    const p_fin = sigmoid(m_fin);
                    const p_xg = (1 - p_block) * p_acc * p_fin;

                    Z_block[idx] = p_block;
                    Z_acc[idx] = p_acc;
                    Z_fin[idx] = p_fin;
                    Z_xg[idx] = p_xg;
                }
            }
            return [Z_block, Z_acc, Z_fin, Z_xg];
        } catch (e) {
            console.error("Predict Error:", e);
            throw e;
        }
    }

    function init() {
        const inputDiv = document.getElementById('inputs-container');
        let presetWrap = document.createElement('div');
        presetWrap.className = 'ctrl-group';
        presetWrap.innerHTML = `<legend>Scenario Presets</legend>
            <select id="preset-select" onchange="applyPreset(this.value)">
                <option value="">-- Select a Scenario --</option>
                ${Object.keys(MODEL.presets).map(k => `<option value="${k}">${k}</option>`).join('')}
            </select>`;
        inputDiv.appendChild(presetWrap);

        const groups = {
            'Context': ['game_state', 'relative_game_state', 'is_home', 'score_diff', 'period_number'],
            'Shooter': ['shooter_role', 'shoots_catches', 'shot_type'],
            'Play Info': ['is_rush', 'is_rebound', 'last_event_type', 'speed_from_last_event', 'dist_from_last_event', 'last_event_time_diff']
        };

        const numericRanges = {
            'speed_from_last_event': {min: 0, max: 60, step: 1},
            'dist_from_last_event': {min: 0, max: 100, step: 1},
            'last_event_time_diff': {min: 0.1, max: 20, step: 0.1},
            'score_diff': {min: -5, max: 5, step: 1},
            'period_number': {min: 1, max: 4, step: 1}
        };

        for(const [gname, fields] of Object.entries(groups)) {
            let fs = document.createElement('fieldset');
            fs.className = 'ctrl-group';
            fs.innerHTML = `<legend>${gname}</legend>`;
            fields.forEach(f => {
                 if (!MODEL.features.includes(f) && !MODEL.options[f] && !numericRanges[f]) return;
                 let wrap = document.createElement('div');
                 wrap.className = 'field';
                 if (numericRanges[f]) {
                    let r = numericRanges[f];
                    wrap.innerHTML = `<label>${f}: <span id="val_${f}"></span></label>`;
                    let sli = document.createElement('input');
                    sli.type = 'range'; sli.id = 'in_' + f;
                    sli.min = r.min; sli.max = r.max; sli.step = r.step;
                    sli.oninput = () => { document.getElementById('val_' + f).innerText = sli.value; updatePlot(); };
                    if (MODEL.defaults[f] !== undefined) sli.value = MODEL.defaults[f];
                    wrap.appendChild(sli);
                 } else if (MODEL.options[f]) {
                    wrap.innerHTML = `<label>${f}</label>`;
                    let sel = document.createElement('select');
                    sel.id = 'in_' + f;
                    sel.onchange = updatePlot;
                    MODEL.options[f].forEach(o => {
                        let opt = document.createElement('option');
                        opt.value = o; opt.innerText = o;
                        sel.appendChild(opt);
                    });
                    if (MODEL.defaults[f] !== undefined) sel.value = MODEL.defaults[f];
                    wrap.appendChild(sel);
                 }
                 fs.appendChild(wrap);
            });
            inputDiv.appendChild(fs);
        }

        Object.keys(numericRanges).forEach(f => {
            let el = document.getElementById('in_' + f);
            if (el) document.getElementById('val_' + f).innerText = el.value;
        });

        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }

    function applyPreset(name) {
        if (!name || !MODEL.presets[name]) return;
        const p = MODEL.presets[name];
        for (const [key, val] of Object.entries(p)) {
            const el = document.getElementById('in_' + key);
            if (el) {
                el.value = val;
                const lbl = document.getElementById('val_' + key);
                if (lbl) lbl.innerText = val;
            }
        }
        updatePlot();
    }

    function getInputs() {
        let inp = {};
        document.querySelectorAll('select, input[type="range"]').forEach(s => {
            if (s.id.startsWith('in_')) inp[s.id.substring(3)] = s.value;
        });
        return inp;
    }

    function convertTo2D(flat) {
        let res = [];
        for(let r=0; r<Y_POINTS; r++) res.push(Array.from(flat.slice(r * X_POINTS, (r + 1) * X_POINTS)));
        return res;
    }

    let plotRevision = 0;
    function updatePlot() {
        try {
            const inputs = getInputs();
            const [zb, za, zf, zxg] = predictScenario(inputs);
            const czb = convertTo2D(zb), cza = convertTo2D(za), czf = convertTo2D(zf), czxg = convertTo2D(zxg);
            let dzb = czb, dza = cza, dzf = czf, dzxg = czxg;
            if (baselineData) {
                dzb = czb.map((row, r) => row.map((val, c) => val - baselineData[0][r][c]));
                dza = cza.map((row, r) => row.map((val, c) => val - baselineData[1][r][c]));
                dzf = czf.map((row, r) => row.map((val, c) => val - baselineData[2][r][c]));
                dzxg = czxg.map((row, r) => row.map((val, c) => val - baselineData[3][r][c]));
            } else {
                dzb = dza = dzf = dzxg = czb.map(r => r.map(c => 0));
            }
            const layout = {
                grid: {rows: 2, columns: 4, pattern: 'independent'},
                paper_bgcolor: '#111', plot_bgcolor: '#111',
                font: {color: 'white', size: 10},
                margin: {t: 60, b: 30, l: 30, r: 30},
                showlegend: false,
                shapes: [],
                datarevision: plotRevision++
            };
            layout.annotations = [
                {text: 'Block Layer', x: 0.1, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#ff5555'}},
                {text: 'Accuracy Layer', x: 0.37, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#55ff55'}},
                {text: 'Finish Layer', x: 0.63, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#5555ff'}},
                {text: 'Final xG Score', x: 0.9, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#ffff55'}},
                {text: 'Δ Block', x: 0.1, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ Accuracy', x: 0.37, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ Finish', x: 0.63, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ xG', x: 0.9, y: 0.48, xref:'paper', yref:'paper', showarrow:false}
            ];
            const traces = [
                {type:'heatmap', x: gridX, y: gridY, z:czb, colorscale:'Magma', zmin:0, zmax:1, xaxis:'x', yaxis:'y', name:'Block', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:cza, colorscale:'Viridis', zmin:0, zmax:1, xaxis:'x2', yaxis:'y2', name:'Acc', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:czf, colorscale:'Viridis', zmin:0, zmax:1, xaxis:'x3', yaxis:'y3', name:'Fin', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:czxg, colorscale:'Hot', zmin:0, zmax:0.4, xaxis:'x4', yaxis:'y4', name:'xG', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:dzb, colorscale:'RdBu', zmid:0, zmin:-0.3, zmax:0.3, xaxis:'x5', yaxis:'y5', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:dza, colorscale:'RdBu', zmid:0, zmin:-0.3, zmax:0.3, xaxis:'x6', yaxis:'y6', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:dzf, colorscale:'RdBu', zmid:0, zmin:-0.3, zmax:0.3, xaxis:'x7', yaxis:'y7', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:dzxg, colorscale:'RdBu', zmid:0, zmin:-0.1, zmax:0.1, xaxis:'x8', yaxis:'y8', zsmooth:'best'}
            ];
            ['','2','3','4','5','6','7','8'].forEach((s) => {
                layout['xaxis'+s] = {range:[0, 100], visible:false, fixedrange:true};
                layout['yaxis'+s] = {range:[-42.5, 42.5], visible:false, scaleanchor:'x'+s, fixedrange:true};
                RINK_SHAPES.forEach(sh => {
                    let sh2 = {...sh}; sh2.xref = 'x' + s; sh2.yref = 'y' + s;
                    layout.shapes.push(sh2);
                });
            });
            Plotly.react('plot', traces, layout);
        } catch (e) {
            console.error("Plot Update Error:", e);
        }
    }

    function setBaseline() {
        const [zb, za, zf, zxg] = predictScenario(getInputs());
        baselineData = [convertTo2D(zb), convertTo2D(za), convertTo2D(zf), convertTo2D(zxg)];
        updatePlot();
    }

    function clearBaseline() { baselineData = null; updatePlot(); }
    window.onload = init;
</script>
</body>
</html>
""".replace('__JSON_DATA__', json_data).replace('__RINK_SHAPES__', rink_shapes_json).replace('__MODEL_NAME__', base_name)
 
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Alternate Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
