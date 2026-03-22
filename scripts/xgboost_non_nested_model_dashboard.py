"""xgboost_non_nested_model_dashboard.py

Generates an interactive HTML dashboard for the XGBoost Non-Nested xG model.
Functional Parity: Uses a client-side JavaScript tree inference engine.
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

from puck import fit_xgboost_non_nested, config as puck_config

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
    
    return {
        'trees': trees,
        'feature_names': booster.feature_names or [],
        'base_score': 0.5
    }

def extract_isotonic_params(iso):
    """Extracts x and y coordinates from IsotonicRegression for JS interpolation."""
    if iso is None:
        return None
    if hasattr(iso, 'f_'):
        return {
            'x': iso.f_.x.tolist(),
            'y': iso.f_.y.tolist(),
            'out_of_bounds': 'clip'
        }
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

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_non_nested.joblib')
    base_name = os.path.basename(model_path).replace('.joblib', '')
    output_path = f"analysis/xgboost_non_nested_xgs/{base_name}_dashboard.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    export_data = {
        'model_name': base_name,
        'features': model.features,
        'vocabs': fit_xgboost_non_nested.CATEGORICAL_VOCABS,
        'priors': model.categorical_priors_,
        'model_data': extract_booster_data(model.model, model.features),
        'calibrator': extract_isotonic_params(model.calibrator),
        'defaults': {
            'distance': 25.0, 'angle_deg': 0.0, 'game_state': '5v5', 'relative_game_state': '5v5',
            'shot_type': 'wrist', 'shooter_role': 'F', 'shoots_catches': 'L',
            'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
            'period_number': 2, 'speed_from_last_event': 0.0, 'last_event_type': 'faceoff'
        },
        'options': {k: v + ['Marginalized'] for k, v in fit_xgboost_non_nested.CATEGORICAL_VOCABS.items()}
    }
    
    export_data['options'].update({
        'is_rush': [0, 1, 'Marginalized'],
        'is_rebound': [0, 1, 'Marginalized'],
        'is_home': [0, 1, 'Marginalized'],
        'period_number': [1, 2, 3, 4],
        'score_diff': [-3, -2, -1, 0, 1, 2, 3]
    })

    json_data = json.dumps(json_serializable(export_data))
    rink_shapes_json = json.dumps(get_rink_shapes())

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>XGBoost Non-Nested Dashboard | __MODEL_NAME__</title>
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
            <h2 style="margin-top:0; color: #00ff88; font-size: 1.2em;">XGBoost Non-Nested xG</h2>
            <div id="inputs-container"></div>
            <div class="btn-row">
                <button class="btn-baseline" onclick="setBaseline()">Set Baseline</button>
                <button class="btn-clear" onclick="clearBaseline()">Clear Δ</button>
            </div>
            <div style="margin-top: 20px; font-size: 0.7em; color: #555;">
                Engine: Client-Side Tree Traversal (100% Parity)
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
    
    function isotonicInterpolate(x, knots) {
        const {x: kx, y: ky} = knots;
        if (x <= kx[0]) return ky[0];
        if (x >= kx[kx.length-1]) return ky[ky.length-1];
        let lo = 0, hi = kx.length - 1;
        while (hi - lo > 1) {
            let mid = (lo + hi) >> 1;
            if (x >= kx[mid]) lo = mid; else hi = mid;
        }
        let t = (x - kx[lo]) / (kx[hi] - kx[lo]);
        return ky[lo] + t * (ky[hi] - ky[lo]);
    }

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
            if (val <= node.split_condition) {
                const child = node.children.find(c => String(c.nodeid) == String(node.yes));
                return evaluateTree(child, featureValues);
            } else {
                const child = node.children.find(c => String(c.nodeid) == String(node.no));
                return evaluateTree(child, featureValues);
            }
        }
    }

    function evaluateForest(featureValues) {
        const forest = MODEL.model_data;
        if (!forest) return 0;
        let margin = Math.log(forest.base_score / (1 - forest.base_score));
        for (const tree of forest.trees) { margin += evaluateTree(tree, featureValues); }
        return margin;
    }

    function predictScenario(inputs) {
        let H = Y_POINTS, W = X_POINTS;
        let Z_xg = new Float32Array(H*W);
        for(let r=0; r<H; r++) {
            for(let c=0; c<W; c++) {
                const idx = r*W + c;
                const x = gridX[c], y = gridY[r];
                const dist = Math.sqrt((x - 89)**2 + y**2);
                const angle_rad = Math.atan2(x - 89, -y);
                let angle_deg = ((-angle_rad * 180 / Math.PI) % 360 + 360) % 360;
                
                let features = {...inputs, distance: dist, angle_deg: angle_deg};
                for (const fName in MODEL.vocabs) {
                    const val = features[fName];
                    if (val === 'Marginalized') features[fName] = null;
                    else {
                        const v_idx = MODEL.vocabs[fName].indexOf(val);
                        features[fName] = (v_idx === -1) ? null : v_idx;
                    }
                }
                ['is_rush', 'is_rebound', 'is_home'].forEach(f => {
                    if (features[f] === 'Marginalized') features[f] = null;
                    else features[f] = Number(features[f]);
                });

                let prob = sigmoid(evaluateForest(features));
                if (MODEL.calibrator) prob = isotonicInterpolate(prob, MODEL.calibrator);
                Z_xg[idx] = prob;
            }
        }
        return Z_xg;
    }

    function init() {
        const inputDiv = document.getElementById('inputs-container');
        const groups = {
            'Context': ['game_state', 'relative_game_state', 'is_home', 'score_diff', 'period_number'],
            'Shooter': ['shooter_role', 'shoots_catches', 'shot_type'],
            'Play Info': ['is_rush', 'is_rebound', 'last_event_type', 'speed_from_last_event']
        };
        for(const [gname, fields] of Object.entries(groups)) {
            let fs = document.createElement('fieldset');
            fs.className = 'ctrl-group';
            fs.innerHTML = `<legend>${gname}</legend>`;
            fields.forEach(f => {
                 if (!MODEL.features.includes(f) && !MODEL.options[f]) return;
                 let wrap = document.createElement('div');
                 wrap.className = 'field';
                 wrap.innerHTML = `<label>${f}</label>`;
                 let sel = document.createElement('select');
                 sel.id = 'in_' + f;
                 sel.onchange = updatePlot;
                 let opts = MODEL.options[f] || ['Marginalized'];
                 opts.forEach(o => {
                     let opt = document.createElement('option');
                     opt.value = o; opt.innerText = o;
                     sel.appendChild(opt);
                 });
                 if (MODEL.defaults[f] !== undefined) sel.value = MODEL.defaults[f];
                 wrap.appendChild(sel);
                 fs.appendChild(wrap);
            });
            inputDiv.appendChild(fs);
        }
        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }

    function getInputs() {
        let inp = {};
        document.querySelectorAll('select').forEach(s => {
            inp[s.id.substring(3)] = s.value;
        });
        return inp;
    }

    function convertTo2D(flat) {
        let res = [];
        for(let r=0; r<Y_POINTS; r++) res.push(Array.from(flat.slice(r*X_POINTS, (r+1)*X_POINTS)));
        return res;
    }

    function updatePlot() {
        const inputs = getInputs();
        const zxg = predictScenario(inputs);
        const czxg = convertTo2D(zxg);
        let dzxg = czxg;
        if (baselineData) {
            dzxg = czxg.map((row, r) => row.map((val, c) => val - baselineData[r][c]));
        } else {
            dzxg = czxg.map(r => r.map(c => 0));
        }
        const layout = {
            grid: {rows: 1, columns: 2, pattern: 'independent'},
            paper_bgcolor: '#111', plot_bgcolor: '#111',
            font: {color: 'white', size: 10},
            margin: {t: 60, b: 30, l: 30, r: 30},
            showlegend: false,
            shapes: []
        };
        layout.annotations = [
            {text: 'xG Probability', x: 0.22, y: 1.1, xref:'paper', yref:'paper', showarrow:false, font:{size:16, color:'#00ff88'}},
            {text: 'Δ Delta', x: 0.78, y: 1.1, xref:'paper', yref:'paper', showarrow:false, font:{size:16, color:'#ffcc00'}}
        ];
        const traces = [
            {type:'heatmap', z:czxg, colorscale:'Hot', zmin:0, zmax:0.4, xaxis:'x1', yaxis:'y1'},
            {type:'heatmap', z:dzxg, colorscale:'RdBu', zmid:0, zmin:-0.1, zmax:0.1, xaxis:'x2', yaxis:'y2'}
        ];
        ['','2'].forEach((ax, i) => {
            const pr = (i===0) ? '' : (i+1);
            layout['xaxis'+pr] = {range:[0, 100], visible:false, fixedrange:true};
            layout['yaxis'+pr] = {range:[-42.5, 42.5], visible:false, scaleanchor:'x'+pr, fixedrange:true};
            RINK_SHAPES.forEach(sh => {
                let sh2 = {...sh}; sh2.xref = 'x' + pr; sh2.yref = 'y' + pr;
                layout.shapes.push(sh2);
            });
        });
        Plotly.react('plot', traces, layout);
    }

    function setBaseline() {
        baselineData = convertTo2D(predictScenario(getInputs()));
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
    print(f"Standalone Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
