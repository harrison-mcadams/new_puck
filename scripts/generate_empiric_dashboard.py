import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, config

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
    print("Loading historical data...")
    # Load all seasons from 2010 onwards
    all_seasons = [str(y) + str(y+1) for y in range(2010, 2027)]
    dfs = []
    for s in all_seasons:
        # Check both root and subdirectory patterns
        fpaths = [
            os.path.join(config.DATA_DIR, f"{s}.csv"),
            os.path.join(config.DATA_DIR, s, f"{s}_df.csv"),
            os.path.join(config.DATA_DIR, s, f"{s}.csv")
        ]
        
        df_season = None
        for fpath in fpaths:
            if os.path.exists(fpath):
                print(f"Reading {fpath}...")
                df_season = pd.read_csv(fpath)
                break
        
        if df_season is not None:
            if 'season' not in df_season.columns:
                df_season['season'] = str(s)
            dfs.append(df_season)
    
    if not dfs:
        print("No data found!")
        return
        
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_raw)} rows.")
    
    # Preprocess
    print("Preprocessing...")
    df = data_pipeline.preprocess_features(
        df_raw, 
        apply_filtering=True, # Filters empty nets, shootouts, non-shots
        apply_imputation=True, # Essential for blocked shots
        verbose=True
    )
    
    print(f"After filtering: {len(df)} rows.")
    
    # Core Features
    features = ['season', 'relative_game_state', 'shot_type', 'shooter_role', 'is_rush', 'is_rebound']
    
    # Fill NAs
    df['season'] = df['season'].astype(str)
    df['relative_game_state'] = df['relative_game_state'].fillna('Unknown')
    df['shot_type'] = df['shot_type'].fillna('Unknown')
    df['shooter_role'] = df['shooter_role'].fillna('Unknown')
    df['is_rush'] = df['is_rush'].fillna(0).astype(int)
    df['is_rebound'] = df['is_rebound'].fillna(0).astype(int)
    
    # Standardize shot types (merge similar or just keep them)
    # Filter only common shot types to keep vocabs manageable
    valid_shots = ['wrist', 'snap', 'slap', 'backhand', 'tip-in', 'wrap-around', 'deflated']
    df['shot_type'] = df['shot_type'].apply(lambda x: x if x in valid_shots else 'Unknown')
    
    valid_roles = ['F', 'D']
    df['shooter_role'] = df['shooter_role'].apply(lambda x: x if x in valid_roles else 'Unknown')
    
    valid_states = ['5v5', '5v4', '4v5', '4v4', '3v3', '5v3', '3v5']
    df['relative_game_state'] = df['relative_game_state'].apply(lambda x: x if x in valid_states else 'Other')

    # Assign Grid Bins
    # 50 points for X (0 to 100), 43 points for Y (-42.5 to 42.5)
    X_POINTS = 50
    Y_POINTS = 43
    df['x_bin'] = np.clip(np.round(df['x'] / 100.0 * (X_POINTS - 1)), 0, X_POINTS - 1).astype(int)
    df['y_bin'] = np.clip(np.round((df['y'] + 42.5) / 85.0 * (Y_POINTS - 1)), 0, Y_POINTS - 1).astype(int)
    
    # Create Outcome Columns
    df['is_block'] = (df['event'] == 'blocked-shot').astype(int)
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    df['attempts'] = 1

    print("Aggregating...")
    grouped = df.groupby(features + ['x_bin', 'y_bin'])[['attempts', 'is_block', 'is_on_net', 'is_goal']].sum().reset_index()
    
    # Create Vocabs
    vocabs = {}
    for f in features:
        vocabs[f] = sorted(grouped[f].unique().tolist())
    
    print("Formatting JSON...")
    # Format data array
    data_array = []
    for row in grouped.itertuples(index=False):
        # Convert feature values to indices
        feat_indices = []
        for i, f in enumerate(features):
            val = getattr(row, f)
            feat_indices.append(vocabs[f].index(val))
            
        data_array.append([
            *feat_indices,
            int(row.x_bin),
            int(row.y_bin),
            int(row.attempts),
            int(row.is_block),
            int(row.is_on_net),
            int(row.is_goal)
        ])

    export_data = {
        'features': features,
        'vocabs': vocabs,
        'data': data_array,
        'x_points': X_POINTS,
        'y_points': Y_POINTS
    }
    
    json_data = json.dumps(export_data)
    rink_shapes_json = json.dumps(get_rink_shapes())
    
    # HTML Template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Empiric Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { margin: 0; background: #111; color: white; font-family: 'Inter', system-ui, -apple-system, sans-serif; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #controls { width: 340px; background: #1a1a1a; padding: 20px; overflow-y: auto; border-right: 1px solid #333; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index: 10; }
        #plot-area { flex: 1; position: relative; background: #111; }
        #plot { width: 100%; height: 100%; }
        .ctrl-group { margin-bottom: 20px; padding: 15px; border: 1px solid #333; border-radius: 8px; background: #222; }
        .ctrl-group legend { padding: 0 10px; font-weight: bold; color: #aaa; font-size: 0.9em; text-transform: uppercase; }
        .field { margin-bottom: 12px; }
        label { display: block; font-size: 0.8em; color: #888; margin-bottom: 4px; }
        
        /* Multi-select styling */
        .multi-select-container { 
            background: #111; 
            border: 1px solid #444; 
            border-radius: 4px; 
            max-height: 150px; 
            overflow-y: auto; 
            padding: 5px;
        }
        .option-row { 
            display: flex; 
            align-items: center; 
            padding: 4px 8px; 
            font-size: 0.85em; 
            cursor: pointer;
            border-radius: 3px;
        }
        .option-row:hover { background: #333; }
        .option-row input { margin-right: 10px; cursor: pointer; }
        .option-row label { display: inline; color: #eee; margin: 0; cursor: pointer; }
        
        .select-actions { display: flex; gap: 10px; margin-top: 5px; }
        .action-link { font-size: 0.7em; color: #00ff88; text-decoration: none; cursor: pointer; opacity: 0.7; }
        .action-link:hover { opacity: 1; }

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
    <div id="loading">Aggregating Data...</div>
    <div id="container">
        <div id="controls">
            <h2 style="margin-top:0; color: #00ff88; font-size: 1.2em;">Empiric Probability Maps</h2>
            <div id="inputs-container"></div>
            
            <fieldset class="ctrl-group">
                <legend>Visualization Settings</legend>
                <div class="field">
                    <label>Smoothing (Sigma): <span id="val_smoothing">2.0</span></label>
                    <input type="range" id="in_smoothing" min="0.1" max="5.0" step="0.1" value="2.0" oninput="document.getElementById('val_smoothing').innerText=this.value; updatePlot();">
                </div>
            </fieldset>

            <div class="btn-row">
                <button class="btn-baseline" onclick="setBaseline()">Set Baseline</button>
                <button class="btn-clear" onclick="clearBaseline()">Clear Δ</button>
            </div>
            
            <div style="margin-top: 20px; font-size: 0.7em; color: #555;">
                Engine: Client-Side Empirical Aggregation + KDE
            </div>
        </div>
        <div id="plot-area">
            <div id="plot"></div>
        </div>
    </div>

<script>
    const DATA = __JSON_DATA__;
    const RINK_SHAPES = __RINK_SHAPES__;
    const X_POINTS = DATA.x_points;
    const Y_POINTS = DATA.y_points;
    
    let gridX = [], gridY = [];
    for(let i=0; i<X_POINTS; i++) gridX.push(i * (100/(X_POINTS-1)));
    for(let i=0; i<Y_POINTS; i++) gridY.push(-42.5 + i * (85/(Y_POINTS-1)));
    
    let baselineData = null;

    function init() {
        const inputDiv = document.getElementById('inputs-container');
        
        let fs = document.createElement('fieldset');
        fs.className = 'ctrl-group';
        fs.innerHTML = `<legend>Core Features</legend>`;
        
        DATA.features.forEach(f => {
            let wrap = document.createElement('div');
            wrap.className = 'field';
            wrap.innerHTML = `<label>${f}</label>`;
            
            let container = document.createElement('div');
            container.className = 'multi-select-container';
            container.id = 'container_' + f;
            
            DATA.vocabs[f].forEach((o, idx) => {
                let row = document.createElement('div');
                row.className = 'option-row';
                
                let cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.id = `cb_${f}_${idx}`;
                cb.value = idx;
                cb.onchange = updatePlot;
                
                // Defaults: 5v5 for relative_game_state, all for others
                if (f === 'relative_game_state') {
                    if (o === '5v5') cb.checked = true;
                } else {
                    cb.checked = true;
                }
                
                let lbl = document.createElement('label');
                lbl.htmlFor = cb.id;
                lbl.innerText = o;
                
                row.onclick = (e) => {
                    if (e.target !== cb && e.target !== lbl) {
                        cb.checked = !cb.checked;
                        updatePlot();
                    }
                };
                
                row.appendChild(cb);
                row.appendChild(lbl);
                container.appendChild(row);
            });
            
            let actions = document.createElement('div');
            actions.className = 'select-actions';
            actions.innerHTML = `
                <a class="action-link" onclick="toggleAll('${f}', true)">All</a>
                <a class="action-link" onclick="toggleAll('${f}', false)">None</a>
            `;
            
            wrap.appendChild(container);
            wrap.appendChild(actions);
            fs.appendChild(wrap);
        });
        inputDiv.appendChild(fs);

        document.getElementById('loading').style.display = 'none';
        updatePlot();
    }

    function toggleAll(feature, state) {
        DATA.vocabs[feature].forEach((o, idx) => {
            document.getElementById(`cb_${feature}_${idx}`).checked = state;
        });
        updatePlot();
    }

    function getInputs() {
        let inp = {};
        DATA.features.forEach(f => {
            let selected = [];
            DATA.vocabs[f].forEach((o, idx) => {
                if (document.getElementById(`cb_${f}_${idx}`).checked) {
                    selected.push(idx);
                }
            });
            inp[f] = selected;
        });
        return inp;
    }

    function blurArray(arr, W, H, sigma) {
        let blurred = new Float32Array(arr.length);
        let radius = Math.ceil(sigma * 3);
        for(let r=0; r<H; r++) {
            for(let c=0; c<W; c++) {
                let sum = 0;
                for(let dr=-radius; dr<=radius; dr++) {
                    for(let dc=-radius; dc<=radius; dc++) {
                        let nr = r + dr;
                        let nc = c + dc;
                        if(nr>=0 && nr<H && nc>=0 && nc<W) {
                            let w = Math.exp(-(dr*dr + dc*dc)/(2*sigma*sigma));
                            sum += arr[nr * W + nc] * w;
                        }
                    }
                }
                blurred[r*W + c] = sum;
            }
        }
        return blurred;
    }

    function convertTo2D(flat) {
        let res = [];
        for(let r=0; r<Y_POINTS; r++) res.push(Array.from(flat.slice(r * X_POINTS, (r + 1) * X_POINTS)));
        return res;
    }

    function calculateScenario(inputs) {
        let grid_att = new Float32Array(X_POINTS * Y_POINTS);
        let grid_blk = new Float32Array(X_POINTS * Y_POINTS);
        let grid_onn = new Float32Array(X_POINTS * Y_POINTS);
        let grid_gol = new Float32Array(X_POINTS * Y_POINTS);
        
        // Filter and aggregate
        for(let i=0; i<DATA.data.length; i++) {
            let row = DATA.data[i];
            let match = true;
            for(let f=0; f<DATA.features.length; f++) {
                let selected = inputs[DATA.features[f]];
                if(!selected.includes(row[f])) {
                    match = false;
                    break;
                }
            }
            if(match) {
                let x_idx = row[DATA.features.length];
                let y_idx = row[DATA.features.length + 1];
                let idx = y_idx * X_POINTS + x_idx;
                grid_att[idx] += row[DATA.features.length + 2];
                grid_blk[idx] += row[DATA.features.length + 3];
                grid_onn[idx] += row[DATA.features.length + 4];
                grid_gol[idx] += row[DATA.features.length + 5];
            }
        }
        
        let sigma = parseFloat(document.getElementById('in_smoothing').value) || 2.0;
        
        let sm_att = blurArray(grid_att, X_POINTS, Y_POINTS, sigma);
        let sm_blk = blurArray(grid_blk, X_POINTS, Y_POINTS, sigma);
        let sm_onn = blurArray(grid_onn, X_POINTS, Y_POINTS, sigma);
        let sm_gol = blurArray(grid_gol, X_POINTS, Y_POINTS, sigma);
        
        let Z_block = new Float32Array(X_POINTS * Y_POINTS);
        let Z_acc = new Float32Array(X_POINTS * Y_POINTS);
        let Z_fin = new Float32Array(X_POINTS * Y_POINTS);
        let Z_xg = new Float32Array(X_POINTS * Y_POINTS);
        
        // Smoothing threshold to avoid crazy outliers in low density areas
        // We require at least '0.1' smoothed attempts
        for(let i=0; i<X_POINTS * Y_POINTS; i++) {
            let att = sm_att[i];
            if(att > 0.05) {
                Z_block[i] = sm_blk[i] / att;
                let unb = att - sm_blk[i];
                Z_acc[i] = unb > 0 ? sm_onn[i] / unb : 0;
                Z_fin[i] = sm_onn[i] > 0 ? sm_gol[i] / sm_onn[i] : 0;
                Z_xg[i] = sm_gol[i] / att;
            } else {
                Z_block[i] = null;
                Z_acc[i] = null;
                Z_fin[i] = null;
                Z_xg[i] = null;
            }
        }
        return [Z_block, Z_acc, Z_fin, Z_xg];
    }

    let plotRevision = 0;
    function updatePlot() {
        try {
            const inputs = getInputs();
            const [zb, za, zf, zxg] = calculateScenario(inputs);
            
            const czb = convertTo2D(zb), cza = convertTo2D(za), czf = convertTo2D(zf), czxg = convertTo2D(zxg);
            let dzb = czb, dza = cza, dzf = czf, dzxg = czxg;
            
            if (baselineData) {
                dzb = czb.map((row, r) => row.map((val, c) => (val === null || baselineData[0][r][c] === null) ? null : val - baselineData[0][r][c]));
                dza = cza.map((row, r) => row.map((val, c) => (val === null || baselineData[1][r][c] === null) ? null : val - baselineData[1][r][c]));
                dzf = czf.map((row, r) => row.map((val, c) => (val === null || baselineData[2][r][c] === null) ? null : val - baselineData[2][r][c]));
                dzxg = czxg.map((row, r) => row.map((val, c) => (val === null || baselineData[3][r][c] === null) ? null : val - baselineData[3][r][c]));
            } else {
                dzb = dza = dzf = dzxg = czb.map(r => r.map(c => null)); // Or 0
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
                {text: 'Block Likelihood', x: 0.1, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#ff5555'}},
                {text: 'On-Net Likelihood', x: 0.37, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#55ff55'}},
                {text: 'Finish Likelihood', x: 0.63, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#5555ff'}},
                {text: 'Total Goal Likelihood', x: 0.9, y: 1.05, xref:'paper', yref:'paper', showarrow:false, font:{size:14, color:'#ffff55'}},
                {text: 'Δ Block', x: 0.1, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ On-Net', x: 0.37, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ Finish', x: 0.63, y: 0.48, xref:'paper', yref:'paper', showarrow:false},
                {text: 'Δ Total Goal', x: 0.9, y: 0.48, xref:'paper', yref:'paper', showarrow:false}
            ];
            
            const traces = [
                {type:'heatmap', x: gridX, y: gridY, z:czb, colorscale:'Magma', zmin:0, zmax:1, xaxis:'x', yaxis:'y', name:'Block', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:cza, colorscale:'Viridis', zmin:0, zmax:1, xaxis:'x2', yaxis:'y2', name:'Acc', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:czf, colorscale:'Viridis', zmin:0, zmax:0.5, xaxis:'x3', yaxis:'y3', name:'Fin', zsmooth:'best'},
                {type:'heatmap', x: gridX, y: gridY, z:czxg, colorscale:'Hot', zmin:0, zmax:0.3, xaxis:'x4', yaxis:'y4', name:'xG', zsmooth:'best'},
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
        const [zb, za, zf, zxg] = calculateScenario(getInputs());
        baselineData = [convertTo2D(zb), convertTo2D(za), convertTo2D(zf), convertTo2D(zxg)];
        updatePlot();
    }

    function clearBaseline() { baselineData = null; updatePlot(); }
    
    window.onload = init;
</script>
</body>
</html>
    """
    html_template = html_template.replace('__JSON_DATA__', json_data)
    html_template = html_template.replace('__RINK_SHAPES__', rink_shapes_json)
    
    out_dir = os.path.join(config.ANALYSIS_DIR, 'empiric_dashboard')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dashboard.html')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print(f"Empiric Dashboard saved to: {out_path}")

if __name__ == "__main__":
    main()
