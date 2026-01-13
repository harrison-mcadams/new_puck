
import sys
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import warnings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, rink, fit_glm_nested

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
    model_path = "analysis/xgs/xg_model_nested.joblib"
    output_path = "analysis/nested_xgs/nested_model_dashboard.html"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    
    # Grid Setup (Coarse Grid to keep file size small)
    # 50x43 points instead of 100x85 reduces size by 4x
    xs = np.linspace(0, 100, 50)
    ys = np.linspace(-42.5, 42.5, 43)
    xx, yy = np.meshgrid(xs, ys)
    grid_df_base = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})
    
    # Default Nuisance Parameters
    grid_df_base['event'] = 'shot-on-goal'
    grid_df_base['period_number'] = 2
    grid_df_base['time_elapsed_in_period_s'] = 600.0
    grid_df_base['total_time_elapsed_s'] = 1800.0
    grid_df_base['last_event_time_diff'] = 10.0
    grid_df_base['period_time_type'] = 'elapsed'
    grid_df_base['home_team_defending_side'] = 'left' 
    grid_df_base['player_name'] = 'Simulated Shooter'
    
    # --- Feature Dimensions ---
    roles = ['F', 'D']
    game_states = ['5v5', '5v4'] 
    shot_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflected', 'wrap-around', 'Unknown']
    handedness = ['L', 'R']
    rush_opts = [0, 1]
    rebound_opts = [0] # Keep rebound simple for now
    
    combinations = list(itertools.product(roles, game_states, shot_types, handedness, rush_opts, rebound_opts))
    print(f"Processing {len(combinations)} scenarios over {len(grid_df_base)} points...")

    # Plot Setup
    subplot_titles = [
        "Block Prob", "Accuracy (Unblocked)", "Finish (On Net)", "Combined xG",
        "Δ Block", "Δ Accuracy", "Δ Finish", "Δ xG"
    ]
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.03
    )

    # Pre-compute Scenarios
    for i, (role, gs, st, hand, is_rush, is_reb) in enumerate(combinations):
        if (i+1) % 10 == 0: print(f"  [{i+1}/{len(combinations)}]", end='\r')
        
        df = grid_df_base.copy()
        df['shooter_role'] = role
        df['game_state'] = gs
        df['shot_type'] = st
        df['shoots_catches'] = hand
        df['is_rush'] = is_rush
        df['is_rebound'] = is_reb
        df['last_event_type'] = 'Faceoff' # Default
        df['score_diff'] = 0
        
        # Preprocess
        processed_df = data_pipeline.preprocess_features(
            df, is_training=False, apply_imputation=False,
            apply_arena_adjustments=False, apply_dithering=False, apply_filtering=False
        )
        
        # Predict All Layers
        p_block = model.predict_proba_layer(processed_df, layer='block')
        p_acc = model.predict_proba_layer(processed_df, layer='accuracy')
        p_finish = model.predict_proba_layer(processed_df, layer='finish')
        p_xg = model.predict_proba(processed_df)[:, 1]
        
        # Reshape
        z_block = p_block.reshape(xx.shape)
        z_acc = p_acc.reshape(xx.shape)
        z_finish = p_finish.reshape(xx.shape)
        z_xg = p_xg.reshape(xx.shape)
        
        # Masking
        # Vectorize the scalar function
        v_rink_height = np.vectorize(rink.rink_half_height_at_x)
        mask = (np.abs(yy) > v_rink_height(xx))
        z_block[mask] = np.nan
        z_acc[mask] = np.nan
        z_finish[mask] = np.nan
        z_xg[mask] = np.nan
        
        scenario_id = f"{role}_{gs}_{st}_{hand}_{is_rush}_{is_reb}"
        
        # Default Visibility
        visible = (role == 'F' and gs == '5v5' and st == 'wrist' and hand == 'L' and is_rush == 0)

        # Add Traces (Row 1)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=z_block.tolist(), coloraxis="coloraxis1", name=scenario_id, visible=visible), row=1, col=1)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=z_acc.tolist(), coloraxis="coloraxis2", name=scenario_id, visible=visible), row=1, col=2)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=z_finish.tolist(), coloraxis="coloraxis2", name=scenario_id, visible=visible), row=1, col=3)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=z_xg.tolist(), coloraxis="coloraxis3", name=scenario_id, visible=visible), row=1, col=4)

    # Add Placeholder Traces for Row 2 (Deltas) - only need 4 total, they don't change per scenario
    # Actually wait, JS needs to update these. 
    # We create 4 separate traces that are ALWAYS visible (or managed by JS)
    # JS will inject data into them.
    
    zeros = np.zeros_like(xx)
    zeros[:] = np.nan
    
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zeros, coloraxis="coloraxis4", name="delta_block", visible=True), row=2, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zeros, coloraxis="coloraxis4", name="delta_acc", visible=True), row=2, col=2)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zeros, coloraxis="coloraxis4", name="delta_finish", visible=True), row=2, col=3)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zeros, coloraxis="coloraxis5", name="delta_xg", visible=True), row=2, col=4)
    
    # Shapes
    all_shapes = []
    for i in range(1, 9):
        # Map linear index to row/col
        # r1: 1,2,3,4. r2: 5,6,7,8 ?? No, Plotly uses 'x1','y1', 'x2','y2' etc.
        # It assigns axes based on subplot creation order.
        # make_subplots(rows=2, cols=4) -> 1..8
        xref = f"x{i}" if i > 1 else "x"
        yref = f"y{i}" if i > 1 else "y"
        all_shapes.extend(get_rink_shapes(xref, yref))

    fig.update_layout(
        title=dict(text="Nested GLM Model Explorer", x=0.5, font=dict(size=24, color='white')),
        width=1800, height=900,
        paper_bgcolor='#111', plot_bgcolor='#111',
        shapes=all_shapes,
        # Color Axes
        coloraxis1=dict(colorscale='Magma', cmin=0, cmax=1.0, colorbar=dict(title="Block", x=0.22, y=0.55, len=0.4, thickness=10)),
        coloraxis2=dict(colorscale='Viridis', cmin=0, cmax=1.0, colorbar=dict(title="Acc/Fin", x=0.74, y=0.55, len=0.4, thickness=10)),
        coloraxis3=dict(colorscale='Plasma', cmin=0, cmax=0.3, colorbar=dict(title="xG", x=1.00, y=0.55, len=0.4, thickness=10)),
        coloraxis4=dict(colorscale='RdBu_r', cmin=-0.2, cmax=0.2, colorbar=dict(title="Δ Component", x=0.74, y=0.0, len=0.4, thickness=10)),
        coloraxis5=dict(colorscale='RdBu_r', cmin=-0.1, cmax=0.1, colorbar=dict(title="Δ xG", x=1.00, y=0.0, len=0.4, thickness=10)),
    )
    
    # Hide axes
    for i in range(1, 9):
        xaxis = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xaxis].update(showgrid=False, zeroline=False, visible=False, range=[0, 100])
        fig.layout[yaxis].update(showgrid=False, zeroline=False, visible=False, range=[-42.5, 42.5], scaleanchor=xaxis.replace('axis', ''))

    # Helper for HTML Options
    def opts(items, labels=None):
        return "".join([f'<option value="{x}">{labels[x] if labels else x}</option>' for x in items])
        
    rush_labels = {0: 'No', 1: 'Yes'}
    reb_labels = {0: 'No', 1: 'Yes'}
    
    controls_html = f"""
    <div style="background:#222; padding:10px; text-align:center; color:white; font-family:sans-serif; display:flex; gap:15px; justify-content:center; align-items:center;">
        <div>Role: <select id="role">{opts(roles)}</select></div>
        <div>State: <select id="gs">{opts(game_states)}</select></div>
        <div>Shot: <select id="st">{opts(shot_types)}</select></div>
        <div>Hand: <select id="hand">{opts(handedness)}</select></div>
        <div>Rush: <select id="rush">{opts(rush_opts, rush_labels)}</select></div>
        <div>Rebound: <select id="reb">{opts(rebound_opts, reb_labels)}</select></div>
        <div style="width:20px;"></div>
        <button id="btn-base" style="background:#28a745; border:none; color:white; padding:5px 10px; cursor:pointer;">Set Baseline</button>
        <button id="btn-rest" style="background:#dc3545; border:none; color:white; padding:5px 10px; cursor:pointer;">Clear</button>
        <span id="lbl-base" style="color:#888; font-size:0.9em;">No Baseline</span>
    </div>
    """
    
    js = """
    <script>
    let baseData = [null, null, null, null]; // Block, Acc, Finish, xG

    function getID() {
        try {
            const ids = ['role', 'gs', 'st', 'hand', 'rush', 'reb'];
            const val = ids.map(x => document.getElementById(x).value).join('_');
            console.log("Generated ID:", val);
            return val;
        } catch (e) {
            console.error("Error generating ID:", e);
            return null;
        }
    }
    
    // Helper to safely clone/diff 2D arrays
    function cloneGrid(grid) {
        if (!grid) return null;
        const H = grid.length;
        if (H === 0) return [];
        if (!grid[0]) return []; // Safety check
        const W = grid[0].length;
        
        // Manual deep copy
        let out = new Array(H);
        for (let r=0; r<H; r++) {
            out[r] = new Array(W);
            if (!grid[r]) continue; // Skip row if missing
            for (let c=0; c<W; c++) {
                out[r][c] = grid[r][c];
            }
        }
        return out;
    }
    
    function makeDeltaGrid(curr, base) {
        if (!curr || !base) return null;
        const H = curr.length;
        if (H === 0) return [];
        if (!curr[0] || !base[0]) return []; // Safety
        const W = curr[0].length;
        
        let out = new Array(H);
        for (let r=0; r<H; r++) {
            out[r] = new Array(W);
            if (!curr[r] || !base[r]) continue;
            for (let c=0; c<W; c++) {
                const A = curr[r][c];
                const B = base[r][c];
                if (A == null || B == null) {
                    out[r][c] = null;
                } else {
                    out[r][c] = A - B;
                }
            }
        }
        return out;
    }
    
    function makeEmptyGrid(fromGrid) {
        if (!fromGrid) return null;
        const H = fromGrid.length;
        if (H === 0) return [];
        if (!fromGrid[0]) return []; // Safety
        const W = fromGrid[0].length;
        let out = new Array(H);
        for (let r=0; r<H; r++) {
            out[r] = new Array(W).fill(null);
        }
        return out;
    }

    function update() {
        const id = getID();
        if (!id) return;
        
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (!gd || !gd.data) {
            console.warn("Graph data not ready");
            return;
        }
        
        // Traces 0..N-5 are the scenarios. Groups of 4.
        // Last 4 traces are Deltas.
        const totalScenarioTraces = gd.data.length - 4;
        
        let vis = new Array(gd.data.length).fill(false);
        let currentData = [null, null, null, null];
        let found = false;
        
        // Find visible group
        for (let i=0; i<totalScenarioTraces; i+=4) {
            if (gd.data[i].name === id) {
                vis[i] = true;   // Block
                vis[i+1] = true; // Acc
                vis[i+2] = true; // Fin
                vis[i+3] = true; // xG
                currentData = [gd.data[i].z, gd.data[i+1].z, gd.data[i+2].z, gd.data[i+3].z];
                found = true;
                break;
            }
        }
        
        if (!found) {
            console.warn("Scenario not found:", id);
            console.log("Total traces:", gd.data.length);
            // Log first 10 trace names to debug
            const names = gd.data.slice(0, 50).map((t, idx) => `${idx}: ${t.name}`);
            console.log("Trace Names (First 50):", names);
        }

        // Show Deltas
        vis[gd.data.length-4] = true;
        vis[gd.data.length-3] = true;
        vis[gd.data.length-2] = true;
        vis[gd.data.length-1] = true;
        
        try {
            Plotly.restyle(gd, {visible: vis});
        } catch (e) {
            console.error("Restyle visibility failed:", e);
        }
        
        // Calculate Deltas
        if (baseData[0] != null && currentData[0] != null) {
            try {
                let deltas = [null, null, null, null];
                for (let k=0; k<4; k++) {
                    deltas[k] = makeDeltaGrid(currentData[k], baseData[k]);
                }
                
                // If invalid result, fallback to empty
                const safeDeltas = deltas.map(d => d ? d : makeEmptyGrid(currentData[0]));

                Plotly.restyle(gd, {z: safeDeltas}, 
                               [gd.data.length-4, gd.data.length-3, gd.data.length-2, gd.data.length-1]);
                               
            } catch (e) {
                console.error("Error calculating/updating deltas:", e);
            }
        } else {
             // Clear deltas
             try {
                 // Try getting shape from current data or first trace
                 const src = currentData[0] || gd.data[0].z;
                 const emptyZ = makeEmptyGrid(src);
                 
                 if (emptyZ) {
                     Plotly.restyle(gd, {z: [emptyZ, emptyZ, emptyZ, emptyZ]}, 
                                    [gd.data.length-4, gd.data.length-3, gd.data.length-2, gd.data.length-1]);
                 }
             } catch (e) {
                 console.error("Error clearing deltas:", e);
             }
        }
    }
    
    document.getElementById('btn-base').onclick = () => {
        const id = getID();
        const gd = document.getElementsByClassName('plotly-graph-div')[0];
        let captured = false;
        
        for (let i=0; i<gd.data.length-4; i+=4) {
            if (gd.data[i].name === id) {
                try {
                    // Manual Deep Clone
                    baseData = [
                        cloneGrid(gd.data[i].z), 
                        cloneGrid(gd.data[i+1].z), 
                        cloneGrid(gd.data[i+2].z), 
                        cloneGrid(gd.data[i+3].z)
                    ];
                    captured = true;
                } catch (e) {
                    console.error("Error cloning baseline data:", e);
                    alert("Error capturing baseline data. Check console.");
                }
                break;
            }
        }
        
        if (captured) {
            document.getElementById('lbl-base').innerText = "Baseline: " + id;
            document.getElementById('lbl-base').style.color = "#28a745";
            console.log("Baseline successfully set for:", id);
            // Force update to show 0 deltas
            update();
        } else {
            console.warn("Could not find trace to set baseline for ID:", id);
            alert("Could not set baseline. Scenario likely not pre-computed or invalid ID.");
        }
    };
    
    document.getElementById('btn-rest').onclick = () => {
        baseData = [null, null, null, null];
        document.getElementById('lbl-base').innerText = "No Baseline";
        document.getElementById('lbl-base').style.color = "#888";
        update();
    }

    const sels = document.querySelectorAll('select');
    sels.forEach(s => s.onchange = update);
    
    // Auto-init
    const check = setInterval(() => {
        if (document.getElementsByClassName('plotly-graph-div')[0]) {
            clearInterval(check);
            update();
        }
    }, 200);
    </script>    
    document.getElementById('btn-rest').onclick = () => {
        baseData = [null, null, null, null];
        document.getElementById('lbl-base').innerText = "No Baseline";
        document.getElementById('lbl-base').style.color = "#888";
        update();
    }

    const sels = document.querySelectorAll('select');
    sels.forEach(s => s.onchange = update);
    
    // Auto-init
    const check = setInterval(() => {
        if (document.getElementsByClassName('plotly-graph-div')[0]) {
            clearInterval(check);
            update();
        }
    }, 200);
    </script>
    """
    
    html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"<html><body style='margin:0; background:#111; color:white;'>{controls_html}{html}{js}</body></html>")
        
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    main()
