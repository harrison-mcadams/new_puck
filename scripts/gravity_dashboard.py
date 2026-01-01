import pandas as pd
import json
import os
import sys

# Paths
DATA_PATH = os.path.join("analysis", "gravity", "player_gravity_season.csv")
OUTPUT_HTML = os.path.join("analysis", "gravity", "gravity_dashboard.html")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gravity Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .controls { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }
        label { font-weight: bold; margin-right: 5px; }
        select, input { padding: 5px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; }
        #plot { width: 100%; height: 80vh; border-radius: 8px; background: #1e1e1e; }
        .metric-value { font-family: monospace; color: #4db8ff; }
    </style>
</head>
<body>

    <div class="controls">
        <div>
            <label for="seasonSelect">Season:</label>
            <select id="seasonSelect" onchange="updatePlot()"></select>
        </div>
        
        <div>
            <label for="teamSelect">Team:</label>
            <select id="teamSelect" onchange="updatePlot()">
                <option value="ALL">All Teams</option>
            </select>
        </div>

        <div style="flex-grow: 1; max-width: 400px;">
            <label for="goalSlider">Min Goals on Ice: <span id="goalValue" class="metric-value">5</span></label>
            <input type="range" id="goalSlider" min="1" max="50" value="5" style="width: 100%;" oninput="document.getElementById('goalValue').innerText = this.value; updatePlot();">
        </div>

        <div>
            <label for="metricSelect">Metric:</label>
            <select id="metricSelect" onchange="updatePlot()">
                <option value="absolute">Relative Gravity (vs League)</option>
                <option value="wowy">Linemate-Adjusted Gravity (vs Line)</option>
            </select>
        </div>
    </div>

    <div id="plot"></div>

    <script>
        // EMBEDDED DATA
        const rawData = __DATA_JSON__;
        
        // Init Controls
        const seasons = [...new Set(rawData.map(d => d.season))].sort();
        const teams = [...new Set(rawData.map(d => d.team_abbr))].sort();
        
        const seasonSelect = document.getElementById('seasonSelect');
        seasons.forEach(s => {
            let opt = document.createElement('option');
            opt.value = s;
            opt.innerText = s;
            seasonSelect.appendChild(opt);
        });
        // Select latest season by default
        seasonSelect.value = seasons[seasons.length - 1];

        const teamSelect = document.getElementById('teamSelect');
        teams.forEach(t => {
            let opt = document.createElement('option');
            opt.value = t;
            opt.innerText = t;
            teamSelect.appendChild(opt);
        });

        function getFilteredData() {
            const s = parseInt(seasonSelect.value);
            const t = teamSelect.value;
            const limit = parseInt(document.getElementById('goalSlider').value);

            return rawData.filter(d => {
                if (d.season !== s) return false;
                if (t !== 'ALL' && d.team_abbr !== t) return false;
                if (d.goals_on_ice_count < limit) return false;
                return true;
            });
        }

        function updatePlot() {
            const data = getFilteredData();
            const metric = document.getElementById('metricSelect').value;
            
            // Split by Position for Colors
            const positions = ['F', 'D'];
            const colors = {'F': '#1f77b4', 'D': '#ff7f0e', 'G': '#2ca02c', 'UNK': '#999'};
            
            const traces = positions.map(pos => {
                const subset = data.filter(d => d.display_position === pos);
                return {
                    x: subset.map(d => d.rel_on_puck_mean_dist_ft),
                    y: subset.map(d => metric === 'absolute' ? d.rel_off_puck_mean_dist_ft : d.rel_to_teammates_off_puck),
                    mode: 'markers',
                    name: pos === 'F' ? 'Forwards' : 'Defense',
                    marker: {
                        size: subset.map(d => d.goals_on_ice_count),
                        sizemode: 'area',
                        sizeref: 2.0 * 50 / (40*40),
                        sizemin: 4,
                        color: colors[pos],
                        opacity: 0.8,
                        line: {width: 1, color: 'white'}
                    },
                    text: subset.map(d => d.player_name),
                    customdata: subset.map(d => [d.team_abbr, d.goals_on_ice_count, d.on_puck_mean_dist_ft, d.rel_off_puck_mean_dist_ft, d.rel_to_teammates_off_puck]),
                    hovertemplate: 
                        "<b>%{text}</b> (%{customdata[0]})<br>" +
                        "Pos: " + pos + "<br>" +
                        "Goals on Ice: %{customdata[1]}<br>" +
                        "Rel On-Puck: %{x:.2f} ft<br>" +
                        "Rel Off-Puck (League): %{customdata[3]:+.2f} ft<br>" +
                        "vs Teammates: %{customdata[4]:+.2f} ft<br>" +
                        "<extra></extra>"
                };
            });

            const layout = {
                template: 'plotly_dark',
                title: metric === 'absolute' ? `Relative Gravity (vs League Baseline)` : `Linemate-Adjusted Gravity (Relative to Line)`,
                xaxis: { title: 'Relative On-Puck MOD (ft) <br><i>(Negative = Draws More Pressure)</i>', autorange: 'reversed' },
                yaxis: { title: metric === 'absolute' ? 'Rel Off-Puck MOD (ft)' : 'Off-Puck Gravity Relative to Teammates (ft)', autorange: 'reversed' },
                hovermode: 'closest',
                shapes: [
                    { type: 'line', x0: 0, y0: -100, x1: 0, y1: 100, line: {color: 'gray', dash: 'dash'} },
                    { type: 'line', x0: -100, y0: 0, x1: 100, y1: 0, line: {color: 'gray', dash: 'dash'} }
                ],
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                font: { color: '#e0e0e0' }
            };

            Plotly.react('plot', traces, layout, {responsive: true});
        }

        // Initial Render
        updatePlot();
    </script>
</body>
</html>
"""

def generate_dashboard():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Cleaning & Mapping
    df = df.dropna(subset=['rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft'])
    if 'team_abbr' not in df.columns:
        df['team_abbr'] = 'UNK'
    
    df['display_position'] = df['position'].map({
        'L': 'F', 'R': 'F', 'C': 'F', 'F': 'F',
        'D': 'D', 'G': 'G'
    }).fillna('UNK')

    # Convert to list of dicts for JSON
    # Keep only necessary columns to reduce file size
    cols = ['season', 'player_name', 'team_abbr', 'display_position', 
            'goals_on_ice_count', 'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft',
            'on_puck_mean_dist_ft', 'off_puck_mean_dist_ft', 'rel_to_teammates_off_puck']
          # Prepare Data
    data_records = df.to_dict(orient='records')
    
    # DEBUG CHECK
    for r in data_records:
        if 'Grebenkin' in str(r.get('player_name')):
             print(f"DEBUG GREBENKIN: {r}")

    json_data = json.dumps(data_records)
    
    # Inject into HTML
    html_content = HTML_TEMPLATE.replace('__DATA_JSON__', json_data)
    
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Interactive dashboard generated: {OUTPUT_HTML} ({len(data_records)} records)")

if __name__ == "__main__":
    generate_dashboard()
