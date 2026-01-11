
import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())
from puck import analyze, features, rink

def main():
    print("--- Generating xG Model Explorer Data ---")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    clf = joblib.load(model_path)

    # 2. Define Scenarios to Pre-calculate
    scenarios = {
        'baseline': {'shot_type': 'wrist', 'game_state': '5v5', 'is_rebound': 0, 'shooter_role': 'F'},
        'slap_shot': {'shot_type': 'slap', 'game_state': '5v5', 'is_rebound': 0, 'shooter_role': 'F'},
        'backhand': {'shot_type': 'backhand', 'game_state': '5v5', 'is_rebound': 0, 'shooter_role': 'F'},
        'defman': {'shot_type': 'wrist', 'game_state': '5v5', 'is_rebound': 0, 'shooter_role': 'D'},
        'power_play': {'shot_type': 'wrist', 'game_state': '5v4', 'is_rebound': 0, 'shooter_role': 'F'},
        'rebound': {'shot_type': 'wrist', 'game_state': '5v5', 'is_rebound': 1, 'shooter_role': 'F', 'rebound_angle_change': 45, 'rebound_time_diff': 1.0},
    }

    # 3. Create Grid (Higher Res for better sampling of sharp drop-offs)
    xs = np.linspace(25, 100, 100) # Attacking zone (0.75 ft steps)
    ys = np.linspace(-42.5, 42.5, 85) # (1.0 ft steps)
    X, Y = np.meshgrid(xs, ys)
    grid_points = pd.DataFrame({'x': X.ravel(), 'y': Y.ravel()})
    
    # Vectorized metrics
    goal_x = 89
    dx = grid_points['x'] - goal_x
    dy = grid_points['y']
    grid_points['distance'] = np.sqrt(dx**2 + dy**2)
    rx, ry = 0.0, -1.0
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    grid_points['angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0
    
    # Default situational features
    grid_points['score_diff'] = 0
    grid_points['period_number'] = 1
    grid_points['time_elapsed_in_period_s'] = 600
    grid_points['total_time_elapsed_s'] = 600
    grid_points['shoots_catches'] = 'L'
    grid_points['is_rush'] = 0
    grid_points['last_event_type'] = 'facetoff'
    grid_points['last_event_time_diff'] = 10
    grid_points['rebound_angle_change'] = 0
    grid_points['rebound_time_diff'] = 0
    grid_points['is_net_empty'] = 0  # Default to goalie in net
    grid_points['event'] = 'shot-on-goal'

    results = {}
    for name, params in scenarios.items():
        print(f"  Calculating {name}...")
        df_temp = grid_points.copy()
        for k, v in params.items():
            df_temp[k] = v
        
        # Predict
        probs = clf.predict_proba(df_temp)[:, 1]
        
        # PHYSICS MASK: Force xG to 0 behind goal line
        # The model may extrapolate high values for angles it never saw (e.g. 270 deg)
        # causing "ghost" hotspots behind the net.
        mask_behind = df_temp['x'] > 89
        probs[mask_behind] = 0.0
        
        if name == 'baseline':
            # Calculate P(Blocked) for visualization
            p_blk = clf.predict_proba_layer(df_temp, 'block')
            p_blk[mask_behind] = 0.0
            z_blk = p_blk.reshape(len(ys), len(xs))
            results['block_prob'] = z_blk.tolist()

        # Reshape to 2D (y, x) for Plotly
        # shape must match (len(ys), len(xs)) -> (85, 100)
        z_2d = probs.reshape(len(ys), len(xs))
        results[name] = z_2d.tolist()

    # 4. Generate HTML
    output_html = 'analysis/nested_xgs/model_explorer.html'
    
    template = """
<!DOCTYPE html>
<html>
<head>
    <title>Nested xG Model Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0b0e14; color: #e0e6ed; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #2e3b4e; padding-bottom: 10px; }
        .controls { background: #1a202c; padding: 20px; border-radius: 12px; margin-bottom: 20px; display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { padding: 10px 18px; border-radius: 6px; border: 1px solid #4a5568; background: #2d3748; color: white; cursor: pointer; transition: 0.2s; font-weight: 500; }
        .btn:hover { background: #4a5568; }
        .btn.active { background: #3182ce; border-color: #63b3ed; box-shadow: 0 0 8px rgba(49, 130, 206, 0.5); }
        #plot { width: 100%; height: 75vh; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
        .info-panel { margin-top: 20px; font-size: 0.9em; color: #a0aec0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Nested xG Model Explorer</h1>
            <div style="text-align: right; font-size: 0.8em; opacity: 0.7;">Trained on 3.3M Shots (2014-2026)</div>
        </div>

        <div class="controls">
            <button class="btn active" onclick="setLayer('baseline', this)">Wrist Shot (5v5 F)</button>
            <button class="btn" onclick="setLayer('slap_shot', this)">Slap Shot</button>
            <button class="btn" onclick="setLayer('backhand', this)">Backhand</button>
            <button class="btn" onclick="setLayer('defman', this)">Defenseman Shot</button>
            <button class="btn" onclick="setLayer('power_play', this)">Power Play (5v4)</button>
            <button class="btn" onclick="setLayer('rebound', this)">🔥 Rebound</button>
            <button class="btn" onclick="setLayer('block_prob', this)">🛡️ Block Prob</button>
        </div>

        <div id="plot"></div>

        <div class="info-panel">
            <b>How to use:</b> Hover over the rink to see localized xG values. Toggle buttons to see how the model "weights" different shot characteristics spatially. 
            <i>Note: "Rebound" assumes a 45-degree angle change from a previous shot 1 second prior.</i>
        </div>
    </div>

    <script>
        const data = __DATA_JSON__;
        const gridX = __GRID_X__;
        const gridY = __GRID_Y__;
        
        let currentLayer = 'baseline';

        function setLayer(layerName, btn) {
            currentLayer = layerName;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updatePlot();
        }

        function updatePlot() {
            const z = data[currentLayer];
            
            const trace = {
                x: gridX,
                y: gridY,
                z: z,
                type: 'contour',
                colorscale: 'Viridis',
                reversescale: false,
                line: { smoothing: 0.85 },
                contours: { coloring: 'heatmap' },
                hovertemplate: 'x: %{x}ft<br>y: %{y}ft<br><b>Val: %{z:.3f}</b><extra></extra>'
            };

            const layout = {
                template: 'plotly_dark',
                paper_bgcolor: '#0b0e14',
                plot_bgcolor: '#0b0e14',
                xaxis: { title: 'Feet from Center', range: [25, 100], fixedrange: true },
                yaxis: { title: 'Width (ft)', range: [-42.5, 42.5], scaleanchor: 'x', scaleratio: 1, fixedrange: true },
                shapes: [
                    // Rink Outline (Board)
                    { type: 'rect', x0: -100, y0: -42.5, x1: 100, y1: 42.5, line: {color: '#4a5568', width: 2}, layer: 'below' },
                    
                    // Center Line (Red)
                    { type: 'line', x0: 0, y0: -42.5, x1: 0, y1: 42.5, line: {color: '#e53e3e', width: 3}, layer: 'below' },
                    
                    // Blue Line (x=25)
                    { type: 'line', x0: 25, y0: -42.5, x1: 25, y1: 42.5, line: {color: '#3182ce', width: 3}, layer: 'below' },
                    
                    // Goal Line (Red, x=89)
                    { type: 'line', x0: 89, y0: -42.5, x1: 89, y1: 42.5, line: {color: '#e53e3e', width: 2} },
                    
                    // Goal Crease (Semi-circle approx as circle for now, simpler in Plotly shapes)
                    { type: 'circle', x0: 83, y0: -6, x1: 95, y1: 6, fillcolor: 'rgba(99, 179, 237, 0.3)', line: {color: '#e53e3e', width: 1}, layer: 'below' },
                    
                    // Faceoff Circle Top (x=69, y=22, r=15 implies x-bounds 54-84, y-bounds 7-37)
                    { type: 'circle', x0: 54, y0: 7, x1: 84, y1: 37, line: {color: '#e53e3e', width: 2}, layer: 'below' },
                    { type: 'circle', x0: 68.5, y0: 21.5, x1: 69.5, y1: 22.5, fillcolor: '#e53e3e', line: {width: 0} }, // Dot
                    
                    // Faceoff Circle Bottom (x=69, y=-22)
                    { type: 'circle', x0: 54, y0: -37, x1: 84, y1: -7, line: {color: '#e53e3e', width: 2}, layer: 'below' },
                    { type: 'circle', x0: 68.5, y0: -22.5, x1: 69.5, y1: -21.5, fillcolor: '#e53e3e', line: {width: 0} }  // Dot
                ],
                margin: { t: 40, b: 60, l: 60, r: 40 }
            };

            Plotly.react('plot', [trace], layout, {responsive: true});
        }

        updatePlot();
    </script>
</body>
</html>
"""
    # Inject Data
    json_data = json.dumps(results)
    grid_x = json.dumps(xs.tolist())
    grid_y = json.dumps(ys.tolist())
    
    html = template.replace('__DATA_JSON__', json_data)
    html = html.replace('__GRID_X__', grid_x)
    html = html.replace('__GRID_Y__', grid_y)
    
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Explorer generated: {output_html}")

if __name__ == "__main__":
    main()
