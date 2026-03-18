def get_html_template():
    """Returns the HTML template as a string."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>F1 Race Predictor & WDC Simulator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #e60012, #000000);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #e60012; text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 40px; }
        .section {
            margin-bottom: 40px; padding: 30px;
            background: #f8f9fa; border-radius: 15px; border-left: 5px solid #e60012;
        }
        h2 { color: #333; margin-bottom: 20px; display: flex; align-items: center; }
        h2:before { content: "🏎️"; margin-right: 10px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #555; font-weight: 600; }
        select, input {
            width: 100%; padding: 12px; border: 2px solid #ddd;
            border-radius: 8px; font-size: 16px; transition: border-color 0.3s;
        }
        select:focus, input:focus { outline: none; border-color: #e60012; }
        button {
            background: #e60012; color: white; padding: 15px 40px; border: none;
            border-radius: 8px; font-size: 18px; cursor: pointer;
            transition: transform 0.2s, background 0.3s; font-weight: 600;
        }
        button:hover { background: #c50010; transform: translateY(-2px); }
        button:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .result {
            margin-top: 20px; padding: 20px; background: white;
            border-radius: 10px; border: 2px solid #e60012;
        }
        .result h3 { color: #e60012; margin-bottom: 10px; }
        .result p { color: #333; font-size: 18px; line-height: 1.6; }
        .loading { display: none; text-align: center; color: #e60012; font-size: 18px; margin-top: 20px; }
        .loading.active { display: block; }
        .stats {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-top: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #e60012, #c50010);
            color: white; padding: 20px; border-radius: 10px; text-align: center;
        }
        .stat-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .stat-label { font-size: 0.9em; opacity: 0.9; }
        .driver-list { list-style: none; padding: 0; }
        .driver-list li {
            padding: 12px; margin: 8px 0; background: #f8f9fa;
            border-radius: 5px; display: flex; justify-content: space-between; align-items: center;
        }
        .driver-list li:nth-child(1) { background: linear-gradient(135deg, #FFD700, #FFA500); font-weight: bold; font-size: 1.1em; }
        .driver-list li:nth-child(2) { background: linear-gradient(135deg, #C0C0C0, #A9A9A9); font-weight: bold; }
        .driver-list li:nth-child(3) { background: linear-gradient(135deg, #CD7F32, #8B4513); font-weight: bold; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab {
            padding: 12px 24px; background: #ddd; border: none; border-radius: 8px;
            cursor: pointer; font-size: 16px; font-weight: 600; transition: all 0.3s;
        }
        .tab:hover { background: #ccc; }
        .tab.active { background: #e60012; color: white; }
        .tab.refresh-btn { background: #28a745; color: white; margin-left: auto; }
        .tab.refresh-btn:hover { background: #218838; }
        .tab.refresh-btn:disabled { background: #6c757d; cursor: not-allowed; }
        .tab.save-btn { background: #0d6efd; color: white; }
        .tab.save-btn:hover { background: #0b5ed7; }
        .tab.save-btn:disabled { background: #6c757d; cursor: not-allowed; }
        .tab.save-btn.saved { background: #4CAF50; }
        .race-ended-badge {
            display: inline-block; background: #dc3545; color: white;
            padding: 4px 10px; border-radius: 5px; font-size: 13px;
            margin-left: 8px; vertical-align: middle;
        }
        .error { background: #ff4444; color: white; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .success { background: #4CAF50; color: white; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .info-badge {
            display: inline-block; background: #17a2b8; color: white;
            padding: 5px 10px; border-radius: 5px; font-size: 14px; margin-left: 10px;
        }
        .save-notice {
            margin-top: 12px; padding: 10px 16px; background: #e8f5e9;
            border: 1px solid #a5d6a7; border-radius: 8px;
            color: #2e7d32; font-size: 14px; display: none;
        }
        .save-notice.visible { display: block; }
    </style>
</head>
<body>
<div class="container">
    <h1>🏁 Formula 1 Prediction System</h1>
    <p class="subtitle">ML Race Predictor &amp; Monte Carlo WDC Simulator — 2025 Season</p>

    <div class="section">
        <h2>Upcoming Race Winner Predictions</h2>
        <p style="margin-bottom:20px;color:#666">AI-powered predictions for remaining races</p>
        <div class="tabs">
            <button class="tab active" onclick="showRace('Las Vegas Grand Prix')">🎰 Las Vegas GP</button>
            <button class="tab" onclick="showRace('Qatar Grand Prix')">🏜️ Qatar GP</button>
            <button class="tab" onclick="showRace('Abu Dhabi Grand Prix')">🏝️ Abu Dhabi GP</button>
            <button class="tab refresh-btn" id="refreshBtn" onclick="refreshCurrentRace()">🔄 Generate New Predictions</button>
            <button class="tab save-btn" id="saveBtn" onclick="saveCurrentPrediction()" disabled>💾 Save Prediction</button>
        </div>
        <div id="saveNotice" class="save-notice"></div>
        <div class="loading" id="raceLoading">⏳ Calculating predictions...</div>
        <div id="raceWinnerResult"></div>
    </div>

    <div class="section">
        <h2>Individual Driver Position Predictor</h2>
        <div class="form-group">
            <label>Select Race:</label>
            <select id="race">
                <option value="Las Vegas Grand Prix">🎰 Las Vegas Grand Prix</option>
                <option value="Qatar Grand Prix">🏜️ Qatar Grand Prix</option>
                <option value="Abu Dhabi Grand Prix">🏝️ Abu Dhabi Grand Prix</option>
            </select>
        </div>
        <div class="form-group">
            <label>Driver:</label>
            <select id="driver">
                {% for driver in drivers %}<option value="{{ driver }}">{{ driver }}</option>{% endfor %}
            </select>
        </div>
        <div class="form-group">
            <label>Team:</label>
            <select id="team">
                {% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}
            </select>
        </div>
        <div class="form-group">
            <label>Starting Grid Position (1–20):</label>
            <input type="number" id="grid" min="1" max="20" value="5">
        </div>
        <button onclick="predictDriver()">Predict Finish Position</button>
        <div id="raceResult"></div>
    </div>

    <div class="section">
        <h2>WDC Championship Simulator</h2>
        <p style="margin-bottom:20px;color:#666">
            Simulations run in the background — the page stays responsive while computing.
        </p>
        <div class="form-group">
            <label>Driver:</label>
            <select id="wdcDriver">
                {% for driver in drivers %}<option value="{{ driver }}" {% if driver == 'NOR' %}selected{% endif %}>{{ driver }}</option>{% endfor %}
            </select>
        </div>
        <div class="form-group">
            <label>Simulations:</label>
            <input type="number" id="simulations" value="1000000" min="1000" max="10000000" step="1000">
        </div>
        <div class="form-group">
            <label>Seed (optional — leave blank for random):</label>
            <input type="number" id="seed" placeholder="e.g. 42">
        </div>
        <button id="wdcButton" onclick="startWDC()">Run Championship Simulation</button>
        <div class="loading" id="wdcLoading">⏳ Running simulations in background…</div>
        <div id="wdcResult"></div>
    </div>
</div>

<script>
const predCache = {};
let currentRace = 'Las Vegas Grand Prix';

// ── Race winner predictions ──────────────────────────────────────────────────

function showRace(name) {
    currentRace = name;
    document.querySelectorAll('.tab:not(.refresh-btn):not(.save-btn)').forEach(t => {
        t.classList.toggle('active', t.textContent.includes(name.split(' ')[0]));
    });
    resetSaveBtn();
    hideSaveNotice();

    if (predCache[name]) {
        displayPredictions(predCache[name]);
        updateRefreshBtn(predCache[name].race_ended);
    } else {
        fetchPredictions(name, false);
    }
}

async function fetchPredictions(name, forceRefresh) {
    document.getElementById('raceLoading').classList.add('active');
    document.getElementById('raceWinnerResult').innerHTML = '';
    resetSaveBtn();
    hideSaveNotice();

    try {
        const res = await fetch('/predict_race_winner', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({race_name: name, force_refresh: forceRefresh}),
        });
        const data = await res.json();
        document.getElementById('raceLoading').classList.remove('active');

        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        if (!data.predictions || !data.predictions.length) {
            document.getElementById('raceWinnerResult').innerHTML =
                '<div class="error">No predictions returned.</div>';
            return;
        }

        predCache[name] = data;
        updateRefreshBtn(data.race_ended);
        displayPredictions(data);
    } catch (err) {
        document.getElementById('raceLoading').classList.remove('active');
        document.getElementById('raceWinnerResult').innerHTML =
            `<div class="error">Error: ${err.message}</div>`;
    }
}

function displayPredictions(data) {
    const top = data.predictions.slice(0, 10);
    const endedBadge = data.race_ended
        ? '<span class="race-ended-badge">🏁 Race Ended</span>'
        : '';

    let html = `<div class="result">
        <h3>🏆 ${data.race_name} — Predicted Results ${endedBadge}</h3>
        <span class="info-badge">${data.cached ? '📋 Cached' : '✨ Fresh'}</span>
        <ul class="driver-list">`;
    top.forEach((p, i) => {
        const m = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `P${p.predicted_position}`;
        html += `<li><span><strong>${m}</strong> ${p.driver} (${p.team})</span>
                     <span>${p.predicted_points} pts | ${p.win_probability}% win</span></li>`;
    });
    html += `</ul>
        <div class="success" style="margin-top:20px">
            <strong>🏆 Predicted Winner:</strong> ${data.winner.driver} — ${data.winner.team}<br>
            <strong>Win probability:</strong> ${data.winner.win_probability}%
        </div></div>`;
    document.getElementById('raceWinnerResult').innerHTML = html;

    // Enable Save button whenever predictions are showing
    const saveBtn = document.getElementById('saveBtn');
    saveBtn.disabled = false;
    saveBtn.classList.remove('saved');
    saveBtn.textContent = '💾 Save Prediction';
}

async function refreshCurrentRace() {
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating…';
    resetSaveBtn();
    hideSaveNotice();

    await fetch('/clear_cache', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({race_name: currentRace}),
    });
    delete predCache[currentRace];
    await fetchPredictions(currentRace, true);

    btn.disabled = false;
    btn.textContent = '🔄 Generate New Predictions';
}

function updateRefreshBtn(ended) {
    const btn = document.getElementById('refreshBtn');
    // Race ended — button stays active so user can keep predicting,
    // but we show a subtle label change to inform them.
    if (ended) {
        btn.disabled = false;
        btn.textContent = '🔄 Re-predict (Race Ended)';
        btn.style.background = '#e67e22';
    } else {
        btn.disabled = false;
        btn.textContent = '🔄 Generate New Predictions';
        btn.style.background = '';
    }
}

// ── Save prediction ──────────────────────────────────────────────────────────

async function saveCurrentPrediction() {
    const prediction = predCache[currentRace];
    if (!prediction) return;

    const saveBtn = document.getElementById('saveBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = '⏳ Saving…';

    try {
        const res = await fetch('/save_prediction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({race_name: currentRace, prediction}),
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);

        saveBtn.classList.add('saved');
        saveBtn.textContent = '✅ Saved!';

        // Update badge in result to show it's now cached
        predCache[currentRace].cached = true;
        showSaveNotice(`Prediction for ${currentRace} saved to server cache.`);

        // Re-enable after 3s so they can save again if they re-predict
        setTimeout(() => {
            saveBtn.disabled = false;
            saveBtn.classList.remove('saved');
            saveBtn.textContent = '💾 Save Prediction';
        }, 3000);

    } catch (err) {
        saveBtn.disabled = false;
        saveBtn.textContent = '💾 Save Prediction';
        showSaveNotice(`Save failed: ${err.message}`, true);
    }
}

function resetSaveBtn() {
    const saveBtn = document.getElementById('saveBtn');
    saveBtn.disabled = true;
    saveBtn.classList.remove('saved');
    saveBtn.textContent = '💾 Save Prediction';
}

function showSaveNotice(msg, isError = false) {
    const el = document.getElementById('saveNotice');
    el.textContent = msg;
    el.classList.add('visible');
    el.style.background = isError ? '#fdecea' : '#e8f5e9';
    el.style.borderColor = isError ? '#ef9a9a' : '#a5d6a7';
    el.style.color = isError ? '#c62828' : '#2e7d32';
    setTimeout(() => el.classList.remove('visible'), 5000);
}

function hideSaveNotice() {
    document.getElementById('saveNotice').classList.remove('visible');
}

// ── Individual driver predict ────────────────────────────────────────────────

async function predictDriver() {
    const driver = document.getElementById('driver').value;
    const team   = document.getElementById('team').value;
    const grid   = document.getElementById('grid').value;
    const race   = document.getElementById('race').value;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({driver, team, grid, race}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        const pts = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1};
        document.getElementById('raceResult').innerHTML = `
            <div class="result">
                <h3>Prediction — ${race}</h3>
                <p><strong>${driver}</strong> (${team}) from <strong>P${grid}</strong></p>
                <div class="success" style="margin-top:15px;font-size:24px">
                    Predicted finish: <strong>P${data.predicted_position}</strong>
                </div>
                <p style="margin-top:15px;color:#666">Points: ${pts[data.predicted_position] || 0}</p>
            </div>`;
    } catch(err) {
        document.getElementById('raceResult').innerHTML =
            `<div class="error">Error: ${err.message}</div>`;
    }
}

// ── WDC simulator (async polling) ────────────────────────────────────────────

let _pollTimer = null;

async function startWDC() {
    const driver  = document.getElementById('wdcDriver').value;
    const sims    = document.getElementById('simulations').value;
    const seedVal = document.getElementById('seed').value.trim();

    document.getElementById('wdcButton').disabled = true;
    document.getElementById('wdcLoading').classList.add('active');
    document.getElementById('wdcResult').innerHTML = '';

    try {
        const body = {wdcDriver: driver, simulations: parseInt(sims)};
        if (seedVal !== '') body.seed = parseInt(seedVal);

        const res  = await fetch('/simulate_wdc', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        _pollTimer = setInterval(() => pollSim(data.job_id, driver, sims), 2000);
    } catch(err) {
        document.getElementById('wdcLoading').classList.remove('active');
        document.getElementById('wdcButton').disabled = false;
        document.getElementById('wdcResult').innerHTML =
            `<div class="error">Error: ${err.message}</div>`;
    }
}

async function pollSim(jobId, driver, sims) {
    try {
        const res  = await fetch(`/sim_status/${jobId}`);
        const data = await res.json();

        if (data.status === 'running') return;

        clearInterval(_pollTimer);
        document.getElementById('wdcLoading').classList.remove('active');
        document.getElementById('wdcButton').disabled = false;

        if (data.status === 'error') {
            document.getElementById('wdcResult').innerHTML =
                `<div class="error">Simulation error: ${data.error}</div>`;
            return;
        }

        document.getElementById('wdcResult').innerHTML = `
            <div class="result">
                <h3>🏆 Championship Simulation Results</h3>
                <p style="margin-bottom:15px;color:#666">
                    Remaining: <em>${data.remaining_races.join(', ')}</em>
                </p>
                <div class="stats">
                    <div class="stat-card"><div class="stat-label">Driver</div><div class="stat-value">${driver}</div></div>
                    <div class="stat-card"><div class="stat-label">Simulations</div><div class="stat-value">${parseInt(sims).toLocaleString()}</div></div>
                    <div class="stat-card"><div class="stat-label">Championships Won</div><div class="stat-value">${data.wins.toLocaleString()}</div></div>
                    <div class="stat-card"><div class="stat-label">WDC Probability</div><div class="stat-value">${data.probability.toFixed(2)}%</div></div>
                    <div class="stat-card"><div class="stat-label">Podium Rate</div><div class="stat-value">${data.stats.podium_rate.toFixed(1)}%</div></div>
                    <div class="stat-card"><div class="stat-label">Top-5 Rate</div><div class="stat-value">${data.stats.top5_rate.toFixed(1)}%</div></div>
                    <div class="stat-card"><div class="stat-label">Avg Final Points</div><div class="stat-value">${Math.round(data.stats.avg_points)}</div></div>
                    <div class="stat-card"><div class="stat-label">Current Points</div><div class="stat-value">${data.stats.current_points}</div></div>
                </div>
            </div>`;
    } catch(err) {
        clearInterval(_pollTimer);
        document.getElementById('wdcLoading').classList.remove('active');
        document.getElementById('wdcButton').disabled = false;
        document.getElementById('wdcResult').innerHTML =
            `<div class="error">Polling error: ${err.message}</div>`;
    }
}

window.onload = () => setTimeout(() => showRace('Las Vegas Grand Prix'), 300);
</script>
</body>
</html>
'''
