"""xgboost_nested_model_dashboard.py

Generates an interactive HTML dashboard for the XGBoost Nested xG model.
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

from puck import fit_xgboost_nested, config as puck_config, data_pipeline

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

def extract_logistic_params(lr):
    """Extracts weights and intercept from LogisticRegression."""
    if lr is None:
        return None
    return {
        'coef': float(lr.coef_[0][0]),
        'intercept': float(lr.intercept_[0])
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
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_nested_20202021.joblib')
    base_name = os.path.basename(model_path).replace('.joblib', '')
    output_path = f"analysis/xgboost_nested_xgs/{base_name}_dashboard.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    # Gather Data
    export_data = {
        'model_name': base_name,
        'features': model.features,
        'vocabs': fit_xgboost_nested.CATEGORICAL_VOCABS,
        'priors': model.categorical_priors_,
        'layers': {
            'block': extract_booster_data(model.model_block, [f for f in model.features if f != 'shot_type']),
            'accuracy': extract_booster_data(model.model_acc, model.features),
            'finish': extract_booster_data(model.model_finish, model.features)
        },
        'calibrators': {
            'block': extract_logistic_params(model.calibrator_block),
            'goal': extract_isotonic_params(model.calibrator_goal)
        },
        'defaults': {
            'distance': 25.0, 'angle_deg': 0.0, 'game_state': '5v5', 'relative_game_state': '5v5',
            'shot_type': 'wrist', 'shooter_role': 'F', 'shoots_catches': 'L',
            'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
            'period_number': 2, 'speed_from_last_event': 0.0, 'last_event_type': 'faceoff'
        },
        'numeric_defaults': data_pipeline.NUMERIC_DEFAULTS,
        'options': {k: v + ['Marginalized'] for k, v in fit_xgboost_nested.CATEGORICAL_VOCABS.items()}
    }

    # Pre-calculate Spatial GLM Grids
    X_POINTS, Y_POINTS = 50, 43
    grid_x = np.linspace(0, 100, X_POINTS)
    grid_y = np.linspace(-42.5, 42.5, Y_POINTS)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_df = pd.DataFrame({'x': gx.flatten(), 'y': gy.flatten()})
    
    if hasattr(model, 'spatial_glm_block_') and model.spatial_glm_block_:
        grid_spatial_layers = {}
        grid_spatial_layers['block'] = model.spatial_glm_block_.predict_proba(grid_df)[:, 1].reshape(Y_POINTS, X_POINTS).tolist()
        grid_spatial_layers['accuracy'] = model.spatial_glm_acc_.predict_proba(grid_df)[:, 1].reshape(Y_POINTS, X_POINTS).tolist()
        grid_spatial_layers['finish'] = model.spatial_glm_fin_.predict_proba(grid_df)[:, 1].reshape(Y_POINTS, X_POINTS).tolist()
        export_data['grid_spatial_layers'] = grid_spatial_layers
    
    # Add numerical options
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
    <title>XGBoost Nested Dashboard | __MODEL_NAME__</title>
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
            <h2 style="margin-top:0; color: #00ff88; font-size: 1.2em;">XGBoost Nested xG</h2>
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
            if (x >= kx[mid]) lo = mid;
            else hi = mid;
        }
        let t = (x - kx[lo]) / (kx[hi] - kx[lo]);
        return ky[lo] + t * (ky[hi] - ky[lo]);
    }
    function evaluateTree(node, featureValues) {
        if (!node) return 0;
        if (node.leaf !== undefined) return node.leaf;
        
        const fName = node.split;
        const val = featureValues[fName];
        
        // Handle Missing
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
            // Numerical split
            if (val <= node.split_condition) {
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

    function getLayerProb(layerName, features) {
        const margin = evaluateForest(layerName, features);
        let prob = sigmoid(margin);
        if (layerName === 'block' && MODEL.calibrators.block) {
            const cal = MODEL.calibrators.block;
            prob = sigmoid(cal.coef * prob + cal.intercept);
        }
        return prob;
    }

    function predictScenario(inputs) {
        try {
            // 1. Common Preprocessing
            let baseFeatures = {...inputs};
            // Categorical Encoding
            for (const fName in MODEL.vocabs) {
                const val = baseFeatures[fName];
                if (val === 'Marginalized') baseFeatures[fName] = null;
                else if (typeof val === 'string') {
                    const v_idx = MODEL.vocabs[fName].indexOf(val);
                    baseFeatures[fName] = (v_idx === -1) ? null : v_idx;
                }
            }
            // Numeric Conversion & Defaults
            MODEL.features.forEach(f => {
                if (baseFeatures[f] === undefined) {
                    baseFeatures[f] = MODEL.numeric_defaults[f] !== undefined ? MODEL.numeric_defaults[f] : 0.0;
                }
                if (baseFeatures[f] !== null && baseFeatures[f] !== undefined && baseFeatures[f] !== 'Marginalized') {
                    if (!MODEL.vocabs[f]) {
                        const num = Number(baseFeatures[f]);
                        if (!isNaN(num)) baseFeatures[f] = num;
                    }
                } else if (baseFeatures[f] === 'Marginalized') {
                    baseFeatures[f] = null;
                }
            });

            let H = Y_POINTS, W = X_POINTS;
            let Z_block = new Float32Array(H*W), Z_acc = new Float32Array(H*W), Z_fin = new Float32Array(H*W), Z_xg = new Float32Array(H*W);
            
            for(let r=0; r<H; r++) {
                for(let c=0; c<W; c++) {
                    const idx = r*W + c;
                    let features = {...baseFeatures};
                    
                    // 1. Resolve Spatial Features
                    const x = gridX[c], y = gridY[r];
                    
                    // Smooth GLM Baselines
                    if (MODEL.grid_spatial_layers) {
                        features['spatial_block'] = MODEL.grid_spatial_layers['block'][r][c];
                        features['spatial_acc'] = MODEL.grid_spatial_layers['accuracy'][r][c];
                        features['spatial_fin'] = MODEL.grid_spatial_layers['finish'][r][c];
                    }
                    
                    // Dynamic Distance/Angle (Secondary Adjustments)
                    const dist = Math.sqrt((x - 89)**2 + y**2);
                    const angle_rad = Math.atan2(x - 89, -y);
                    let angle_deg = ((-angle_rad * 180 / Math.PI) % 360 + 360) % 360;
                    features.distance = dist;
                    features.angle_deg = angle_deg;

                    const m_block = evaluateForest('block', features);
                    const m_acc = evaluateForest('accuracy', features);
                    const m_fin = evaluateForest('finish', features);
                    
                    const p_block = getLayerProb('block', features);
                    const p_acc = sigmoid(m_acc);
                    const p_fin = sigmoid(m_fin);
                    
                    let p_xg = (1 - p_block) * p_acc * p_fin;
                    if (MODEL.calibrators.goal) p_xg = isotonicInterpolate(p_xg, MODEL.calibrators.goal);

                    if (r === 0 && c === 0) {
                        console.log("DEBUG [0,0]:", {
                            inputs: inputs,
                            features: features,
                            margins: {block: m_block, acc: m_acc, fin: m_fin},
                            probs: {block: p_block, acc: p_acc, fin: p_fin},
                            final_xg: p_xg
                        });
                    }

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
        console.log("Initializing UI. Features in MODEL:", MODEL.features);
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
        if (!flat) return [];
        let res = [];
        for(let r=0; r<Y_POINTS; r++) {
            const start = r * X_POINTS;
            const end = (r + 1) * X_POINTS;
            if (end > flat.length) {
                 console.error("Index out of bounds in convertTo2D:", end, flat.length);
                 break;
            }
            res.push(Array.from(flat.slice(start, end)));
        }
        return res;
    }

    let plotRevision = 0;
    function updatePlot() {
        try {
            console.log("updatePlot starting...");
            const inputs = getInputs();
            console.log("Current Inputs:", inputs);
            
            const [zb, za, zf, zxg] = predictScenario(inputs);
            console.log("Prediction Complete. xG First Pixel:", zxg[0]);
            
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
            console.log("Calling Plotly.react...");
            Plotly.react('plot', traces, layout);
            console.log("updatePlot finished.");
        } catch (e) {
            console.error("Plot Update Error:", e);
            alert("Update Error: " + e.message);
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
    print(f"Standalone Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
