"""xgboost_tensor_model_dashboard.py

Generates an interactive HTML dashboard for the XGBoost Alternate xG model.
Evaluates predictions dynamically using the same Python backend prediction routines to prevent drift.
Supports Full Bipartite Marginalization (categorical joint priors and numerical native default paths).
"""

import sys
import os
import joblib
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgboost_tensor, config as puck_config, data_pipeline

def json_serializable(obj):
    import numpy as np
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

def get_rink_shapes(xref='x', yref='y'):
    """Full rink shapes for offensive zone."""
    shapes = []
    line_color = "rgba(255, 255, 255, 0.15)"
    red_line_color = "rgba(255, 0, 0, 0.25)"
    blue_line_color = "rgba(0, 132, 255, 0.25)"
    
    # Outer Boards
    shapes.append(dict(type="line", x0=0, y0=42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=100, y1=-42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    shapes.append(dict(type="line", x0=100, y0=-42.5, x1=100, y1=42.5, xref=xref, yref=yref, line=dict(color=line_color, width=2)))
    
    # Center Red Line and Blue Line
    shapes.append(dict(type="line", x0=0, y0=-42.5, x1=0, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=3)))
    shapes.append(dict(type="line", x0=25, y0=-42.5, x1=25, y1=42.5, xref=xref, yref=yref, line=dict(color=blue_line_color, width=3)))
    
    # Goal Line and Net Crease
    shapes.append(dict(type="line", x0=89, y0=-42.5, x1=89, y1=42.5, xref=xref, yref=yref, line=dict(color=red_line_color, width=2)))
    shapes.append(dict(type="circle", x0=83, y0=-6, x1=95, y1=6, xref=xref, yref=yref, fillcolor="rgba(0, 132, 255, 0.1)", line=dict(color=red_line_color, width=1.5)))
    
    # Faceoff Circles Top & Bottom (r=15, dots at x=69, y=22 and y=-22)
    shapes.append(dict(type="circle", x0=54, y0=7, x1=84, y1=37, xref=xref, yref=yref, line=dict(color=red_line_color, width=1.5)))
    shapes.append(dict(type="circle", x0=68.5, y0=21.5, x1=69.5, y1=22.5, xref=xref, yref=yref, fillcolor=red_line_color, line=dict(width=0)))
    
    shapes.append(dict(type="circle", x0=54, y0=-37, x1=84, y1=-7, xref=xref, yref=yref, line=dict(color=red_line_color, width=1.5)))
    shapes.append(dict(type="circle", x0=68.5, y0=-22.5, x1=69.5, y1=-21.5, xref=xref, yref=yref, fillcolor=red_line_color, line=dict(width=0)))
    
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
        'defaults': {
            'distance': 25.0, 'angle_deg': 0.0, 'game_state': '5v5', 'relative_game_state': '5v5',
            'shot_type': 'wrist', 'shooter_role': 'F', 'shoots_catches': 'L',
            'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
            'period_number': 2, 'speed_from_last_event': 7.5, 'last_event_type': 'giveaway',
            'dist_from_last_event': 15.0, 'last_event_time_diff': 2.0, 'season': 20242025
        },
        'numeric_defaults': data_pipeline.NUMERIC_DEFAULTS,
        'options': {str(k): list(v) + ['Marginalized'] for k, v in fit_xgboost_tensor.CATEGORICAL_VOCABS.items()},
        'presets': {
            'Owen Tippett (Clean Shot)': {
                'x': 78, 'y': 10, 'shot_type': 'wrist', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 0, 'period_number': 2, 'score_diff': 0,
                'last_event_type': 'giveaway', 'last_event_time_diff': 2.0, 'dist_from_last_event': 30.0, 'speed_from_last_event': 15.0
            },
            'Flyers Rebound Goal (Owen Tippett)': {
                'x': 84.0, 'y': 0.0, 'shot_type': 'bat', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 1, 'period_number': 2.0, 'score_diff': 2.0,
                'last_event_type': 'shot-on-goal', 'last_event_time_diff': 2.0, 'dist_from_last_event': 9.22, 'speed_from_last_event': 4.61,
                'rebound_angle_change': 26.57, 'rebound_time_diff': 2.0, 'rebound_source': 'none'
            },
            'Flyers Rebound Goal (Denver Barkey)': {
                'x': 86.0, 'y': 4.0, 'shot_type': 'snap', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 1, 'period_number': 2.0, 'score_diff': 2.0,
                'last_event_type': 'shot-on-goal', 'last_event_time_diff': 2.0, 'dist_from_last_event': 25.50, 'speed_from_last_event': 12.75,
                'rebound_angle_change': 83.71, 'rebound_time_diff': 2.0, 'rebound_source': 'none'
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
            'High-Danger Rebound (Goalie Displaced)': {
                'x': 85, 'y': 2, 'shot_type': 'backhand', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 1, 'period_number': 2, 'score_diff': 0,
                'last_event_type': 'shot-on-goal', 'last_event_time_diff': 0.6, 'dist_from_last_event': 5.0, 'speed_from_last_event': 8.0,
                'rebound_angle_change': 65.0, 'rebound_time_diff': 0.6, 'rebound_source': 'shot-on-goal'
            },
            'Typical Rebound (Goalie Recovered)': {
                'x': 83, 'y': -6, 'shot_type': 'snap', 'game_state': '5v5', 'relative_game_state': '5v5',
                'is_rush': 0, 'is_rebound': 1, 'period_number': 2, 'score_diff': 0,
                'last_event_type': 'shot-on-goal', 'last_event_time_diff': 1.8, 'dist_from_last_event': 8.0, 'speed_from_last_event': 4.0,
                'rebound_angle_change': 15.0, 'rebound_time_diff': 1.8, 'rebound_source': 'shot-on-goal'
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

    json_data = json.dumps(json_serializable(export_data))
    rink_shapes_json = json.dumps(get_rink_shapes())

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>XGBoost Model Explorer Dashboard | __MODEL_NAME__</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { margin: 0; background: #0c0f16; color: white; font-family: 'Inter', system-ui, -apple-system, sans-serif; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #controls { width: 330px; background: #131924; padding: 20px; overflow-y: auto; border-right: 1px solid #232e42; box-shadow: 4px 0 15px rgba(0,0,0,0.6); z-index: 10; }
        #plot-area { flex: 1; position: relative; background: #0c0f16; }
        #plot { width: 100%; height: 100%; }
        .ctrl-group { margin-bottom: 20px; padding: 15px; border: 1px solid #232e42; border-radius: 12px; background: #192130; }
        .ctrl-group legend { padding: 0 10px; font-weight: bold; color: #a0aec0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }
        .field { margin-bottom: 16px; }
        .field-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        label { display: block; font-size: 0.8em; color: #a0aec0; }
        .toggle-container { font-size: 0.8em; color: #718096; cursor: pointer; display: flex; align-items: center; gap: 4px; }
        .toggle-container input[type=checkbox] { cursor: pointer; accent-color: #00ff88; margin: 0; }
        select { width: 100%; background: #232e42; color: white; border: 1px solid #2d3d57; padding: 8px; border-radius: 6px; box-sizing: border-box; font-family: inherit; }
        select:focus { outline: none; border-color: #00ff88; }
        input[type=range] { width: 100%; margin-top: 8px; -webkit-appearance: none; background: #2d3d57; height: 6px; border-radius: 3px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: #00ff88; border-radius: 50%; cursor: pointer; box-shadow: 0 0 5px rgba(0,255,136,0.5); }
        .field-label-row label span { font-weight: bold; color: #00ff88; }
        .field.marginalized input[type=range] { opacity: 0.25; pointer-events: none; }
        .field.marginalized .field-label-row label span { color: #4a5568 !important; text-decoration: line-through; }
        .btn-row { display: flex; gap: 10px; margin-top: 20px; }
        button { flex: 1; padding: 10px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; transition: opacity 0.2s; font-family: inherit; }
        .btn-baseline { background: #2d5a27; color: #fff; }
        .btn-clear { background: #5a2727; color: #fff; }
        #loading { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(12,15,22,0.9); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 1000; font-size: 1.5em; gap: 10px; }
        .spinner { width: 40px; height: 40px; border: 4px solid rgba(0,255,136,0.1); border-top-color: #00ff88; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div>Initializing Prediction Engine...</div>
    </div>
    <div id="container">
        <div id="controls">
            <h2 style="margin-top:0; color: #00ff88; font-size: 1.25em; border-bottom: 1px solid #232e42; padding-bottom: 10px;">XGBoost Model Explorer</h2>
            <div id="inputs-container"></div>
            <div class="btn-row">
                <button class="btn-baseline" onclick="setBaseline()">Set Baseline</button>
                <button class="btn-clear" onclick="clearBaseline()">Clear Δ</button>
            </div>
            <div class="ctrl-group" style="margin-top: 20px;">
                <legend>Color Scale Settings</legend>
                <div class="field">
                    <div class="field-label-row">
                        <label>Max xG Color Limit: <span id="val_max_xg">0.40</span></label>
                    </div>
                    <input type="range" id="in_max_xg" min="0.05" max="1.00" step="0.05" value="0.40" oninput="document.getElementById('val_max_xg').innerText = parseFloat(this.value).toFixed(2); updatePlot();">
                </div>
                <div class="field" style="margin-bottom: 0;">
                    <div class="field-label-row">
                        <label>Max Δ Color Limit: <span id="val_max_delta">0.30</span></label>
                    </div>
                    <input type="range" id="in_max_delta" min="0.01" max="0.50" step="0.01" value="0.30" oninput="document.getElementById('val_max_delta').innerText = parseFloat(this.value).toFixed(2); updatePlot();">
                </div>
            </div>
            <div style="margin-top: 25px; font-size: 0.75em; color: #4a5568; line-height: 1.4;">
                <b>Engine</b>: Dynamic Flask Prediction Backend<br>
                <b>Marginalization</b>: Bipartite (Continuous Numerical + Prior-Weighted Categorical)
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
    let currentRequestId = 0;

    async function predictScenario(inputs) {
        const requestId = ++currentRequestId;
        try {
            const isLocalFile = window.location.protocol === 'file:';
            const predictUrl = isLocalFile ? 'http://localhost:8000/predict_model' : '/predict_model';
            
            const response = await fetch(predictUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: MODEL.model_name,
                    inputs: inputs
                })
            });
            const data = await response.json();
            
            // Abort if a newer request is already in flight
            if (requestId !== currentRequestId) return null;
            
            if (data.error) {
                console.error("Prediction Backend Error:", data.error);
                return null;
            }
            
            return [
                new Float32Array(data.block),
                new Float32Array(data.accuracy),
                new Float32Array(data.finish),
                new Float32Array(data.xg)
            ];
        } catch (e) {
            console.error("Fetch Error:", e);
            return null;
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
            'Context': ['season', 'game_state', 'relative_game_state', 'is_home', 'score_diff', 'period_number'],
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

        // GATHER ALL OTHER FEATURES DYNAMICALLY FROM ACTIVE MODEL
        const predefinedFields = new Set();
        for(const fields of Object.values(groups)) {
            fields.forEach(f => predefinedFields.add(f));
        }
        
        const otherFields = [];
        MODEL.features.forEach(f => {
            if (!predefinedFields.has(f) && f !== 'distance' && f !== 'angle_deg' && f !== 'x' && f !== 'y') {
                otherFields.push(f);
            }
        });
        
        if (otherFields.length > 0) {
            groups['Other Features'] = otherFields;
        }

        // Dynamically add default numeric ranges for other numeric fields
        otherFields.forEach(f => {
            if (!MODEL.options[f] && !numericRanges[f]) {
                const defVal = MODEL.defaults[f] !== undefined ? MODEL.defaults[f] : 0.0;
                let min = 0, max = 100, step = 1;
                if (f.includes('time') || f.includes('diff') || f.includes('speed')) {
                    min = 0; max = 50; step = 0.5;
                } else if (f.includes('dist')) {
                    min = 0; max = 150; step = 1;
                } else if (f.includes('angle')) {
                    min = 0; max = 180; step = 1;
                } else if (defVal < 0) {
                    min = -10; max = 10; step = 0.1;
                } else if (defVal <= 1.0) {
                    min = 0; max = 1.0; step = 0.05;
                }
                numericRanges[f] = {min: min, max: max, step: step};
            }
        });

        // BUILD RENDERED FIELDS
        for(const [gname, fields] of Object.entries(groups)) {
            let fs = document.createElement('fieldset');
            fs.className = 'ctrl-group';
            fs.innerHTML = `<legend>${gname}</legend>`;
            
            fields.forEach(f => {
                 if (!MODEL.features.includes(f) && !MODEL.options[f] && !numericRanges[f]) return;
                 let wrap = document.createElement('div');
                 wrap.className = 'field';
                 wrap.id = 'field_' + f;
                 
                 if (numericRanges[f]) {
                    let r = numericRanges[f];
                    const defVal = MODEL.defaults[f] !== undefined ? MODEL.defaults[f] : (r.min + r.max) / 2;
                    wrap.innerHTML = `<div class="field-label-row">
                        <label>${f}: <span id="val_${f}"></span></label>
                        <label class="toggle-container">
                            <input type="checkbox" id="chk_${f}" onchange="toggleMarginalize('${f}')"> Auto
                        </label>
                    </div>`;
                    
                    let sli = document.createElement('input');
                    sli.type = 'range'; sli.id = 'in_' + f;
                    sli.min = r.min; sli.max = r.max; sli.step = r.step; sli.value = defVal;
                    sli.oninput = () => { onSliderInput(f); };
                    wrap.appendChild(sli);
                 } else if (MODEL.options[f]) {
                    wrap.innerHTML = `<div class="field-label-row"><label>${f}</label></div>`;
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

        // Initialize slider numerical text badges
        Object.keys(numericRanges).forEach(f => {
            let el = document.getElementById('in_' + f);
            if (el) document.getElementById('val_' + f).innerText = el.value;
        });

        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }

    function toggleMarginalize(f) {
        const wrap = document.getElementById('field_' + f);
        const chk = document.getElementById('chk_' + f);
        const sli = document.getElementById('in_' + f);
        const valSpan = document.getElementById('val_' + f);
        
        if (chk.checked) {
            wrap.classList.add('marginalized');
            valSpan.innerText = 'Auto';
        } else {
            wrap.classList.remove('marginalized');
            valSpan.innerText = sli.value;
        }
        updatePlot();
    }

    function onSliderInput(f) {
        const sli = document.getElementById('in_' + f);
        const valSpan = document.getElementById('val_' + f);
        valSpan.innerText = sli.value;
        updatePlot();
    }

    function applyPreset(name) {
        if (!name || !MODEL.presets[name]) return;
        const p = MODEL.presets[name];
        for (const [key, val] of Object.entries(p)) {
            const el = document.getElementById('in_' + key);
            const chk = document.getElementById('chk_' + key);
            if (el) {
                if (chk) {
                    if (val === 'Marginalized') {
                        chk.checked = true;
                        document.getElementById('field_' + key).classList.add('marginalized');
                        document.getElementById('val_' + key).innerText = 'Auto';
                    } else {
                        chk.checked = false;
                        document.getElementById('field_' + key).classList.remove('marginalized');
                        el.value = val;
                        document.getElementById('val_' + key).innerText = val;
                    }
                } else {
                    el.value = val;
                }
            }
        }
        updatePlot();
    }

    function getInputs() {
        let inp = {};
        // Read selects
        document.querySelectorAll('select').forEach(s => {
            if (s.id.startsWith('in_')) inp[s.id.substring(3)] = s.value;
        });
        // Read range sliders (supporting numerical marginalization checked state)
        document.querySelectorAll('input[type="range"]').forEach(s => {
            if (s.id.startsWith('in_')) {
                const f = s.id.substring(3);
                const chk = document.getElementById('chk_' + f);
                if (chk && chk.checked) {
                    inp[f] = 'Marginalized';
                } else {
                    inp[f] = s.value;
                }
            }
        });
        return inp;
    }

    function convertTo2D(flat) {
        let res = [];
        for(let r=0; r<Y_POINTS; r++) res.push(Array.from(flat.slice(r * X_POINTS, (r + 1) * X_POINTS)));
        return res;
    }

    let plotRevision = 0;
    async function updatePlot() {
        try {
            const inputs = getInputs();
            const preds = await predictScenario(inputs);
            if (!preds) return; // Aborted due to newer request or error
            
            const maxXg = parseFloat(document.getElementById('in_max_xg')?.value || 0.40);
            const maxDelta = parseFloat(document.getElementById('in_max_delta')?.value || 0.30);
            
            const [zb, za, zf, zxg] = preds;
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
                paper_bgcolor: '#0c0f16', plot_bgcolor: '#0c0f16',
                font: {color: 'white', size: 10},
                margin: {t: 60, b: 30, l: 30, r: 85},
                showlegend: false,
                shapes: [],
                datarevision: plotRevision++
            };
            
            layout.annotations = [
                {text: 'Block Layer', x: 0.1, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:13, color:'#ff5555', bold:true}},
                {text: 'Accuracy Layer', x: 0.37, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:13, color:'#00ff88', bold:true}},
                {text: 'Finish Layer', x: 0.63, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:13, color:'#00aaff', bold:true}},
                {text: 'Final xG Score', x: 0.9, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:13, color:'#ffff55', bold:true}},
                {text: 'Δ Block', x: 0.1, y: 0.48, xref:'paper', yref:'paper', showarrow:false, font:{color: '#a0aec0'}},
                {text: 'Δ Accuracy', x: 0.37, y: 0.48, xref:'paper', yref:'paper', showarrow:false, font:{color: '#a0aec0'}},
                {text: 'Δ Finish', x: 0.63, y: 0.48, xref:'paper', yref:'paper', showarrow:false, font:{color: '#a0aec0'}},
                {text: 'Δ xG', x: 0.9, y: 0.48, xref:'paper', yref:'paper', showarrow:false, font:{color: '#a0aec0'}}
            ];
            
            const traces = [
                {type:'heatmap', x: gridX, y: gridY, z:czb, colorscale:'Magma', zmin:0, zmax:1, xaxis:'x', yaxis:'y', name:'Block', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:cza, colorscale:'Viridis', zmin:0, zmax:1, xaxis:'x2', yaxis:'y2', name:'Acc', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:czf, colorscale:'Viridis', zmin:0, zmax:1, xaxis:'x3', yaxis:'y3', name:'Fin', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:czxg, colorscale:'Hot', zmin:0, zmax:maxXg, xaxis:'x4', yaxis:'y4', name:'xG', zsmooth:'best', showscale:true, colorbar:{title:'xG', thickness:15, len:0.35, y:0.75, x:1.02}},
                {type:'heatmap', x: gridX, y: gridY, z:dzb, colorscale:'RdBu', zmid:0, zmin:-maxDelta, zmax:maxDelta, xaxis:'x5', yaxis:'y5', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:dza, colorscale:'RdBu', zmid:0, zmin:-maxDelta, zmax:maxDelta, xaxis:'x6', yaxis:'y6', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:dzf, colorscale:'RdBu', zmid:0, zmin:-maxDelta, zmax:maxDelta, xaxis:'x7', yaxis:'y7', zsmooth:'best', showscale:false},
                {type:'heatmap', x: gridX, y: gridY, z:dzxg, colorscale:'RdBu', zmid:0, zmin:-maxDelta/3.0, zmax:maxDelta/3.0, xaxis:'x8', yaxis:'y8', zsmooth:'best', showscale:true, colorbar:{title:'Δ xG', thickness:15, len:0.35, y:0.25, x:1.02}}
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

    async function setBaseline() {
        const inputs = getInputs();
        const preds = await predictScenario(inputs);
        if (!preds) return;
        
        baselineData = [convertTo2D(preds[0]), convertTo2D(preds[1]), convertTo2D(preds[2]), convertTo2D(preds[3])];
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
    print(f"Alternate Dynamic Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
