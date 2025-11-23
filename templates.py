def get_html_template():
    """Returns the HTML template as a string"""
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
        h1 {
            color: #e60012;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
        }
        .section {
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #e60012;
        }
        h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        h2:before {
            content: "🏎️";
            margin-right: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
        }
        select, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #e60012;
        }
        button {
            background: #e60012;
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            transition: transform 0.2s, background 0.3s;
            font-weight: 600;
        }
        button:hover {
            background: #c50010;
            transform: translateY(-2px);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            border: 2px solid #e60012;
        }
        .result h3 {
            color: #e60012;
            margin-bottom: 10px;
        }
        .result p {
            color: #333;
            font-size: 18px;
            line-height: 1.6;
        }
        .loading {
            display: none;
            text-align: center;
            color: #e60012;
            font-size: 18px;
            margin-top: 20px;
        }
        .loading.active {
            display: block;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #e60012, #c50010);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .driver-list {
            list-style: none;
            padding: 0;
        }
        .driver-list li {
            padding: 12px;
            margin: 8px 0;
            background: #f8f9fa;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .driver-list li:nth-child(1) { 
            background: linear-gradient(135deg, #FFD700, #FFA500); 
            font-weight: bold;
            font-size: 1.1em;
        }
        .driver-list li:nth-child(2) { 
            background: linear-gradient(135deg, #C0C0C0, #A9A9A9); 
            font-weight: bold; 
        }
        .driver-list li:nth-child(3) { 
            background: linear-gradient(135deg, #CD7F32, #8B4513); 
            font-weight: bold; 
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 12px 24px;
            background: #ddd;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .tab:hover {
            background: #ccc;
        }
        .tab.active {
            background: #e60012;
            color: white;
        }
        .error {
            background: #ff4444;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .success {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏁 Formula 1 Prediction System</h1>
        <p class="subtitle">ML Race Predictor & Monte Carlo WDC Simulator - 2025 Season</p>
        
        <div class="section">
            <h2>Upcoming Race Winner Predictions</h2>
            <p style="margin-bottom: 20px; color: #666;">AI-powered predictions for the remaining races</p>
            
            <div class="tabs">
                <button class="tab active" onclick="predictSpecificRace('Las Vegas Grand Prix')">🎰 Las Vegas GP</button>
                <button class="tab" onclick="predictSpecificRace('Qatar Grand Prix')">🏜️ Qatar GP</button>
                <button class="tab" onclick="predictSpecificRace('Abu Dhabi Grand Prix')">🏝️ Abu Dhabi GP</button>
            </div>
            
            <div class="loading" id="raceLoading">
                ⏳ Calculating predictions...
            </div>
            <div id="raceWinnerResult"></div>
        </div>
        
        <div class="section">
            <h2>Individual Driver Position Predictor</h2>
            <p style="margin-bottom: 20px; color: #666;">Predict a specific driver's finishing position for upcoming races</p>
            
            <form id="raceForm">
                <div class="form-group">
                    <label>Select Race:</label>
                    <select id="race" name="race" required>
                        <option value="Las Vegas Grand Prix">🎰 Las Vegas Grand Prix</option>
                        <option value="Qatar Grand Prix">🏜️ Qatar Grand Prix</option>
                        <option value="Abu Dhabi Grand Prix">🏝️ Abu Dhabi Grand Prix</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Driver:</label>
                    <select id="driver" name="driver" required>
                        {% for driver in drivers %}
                        <option value="{{ driver }}">{{ driver }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Team:</label>
                    <select id="team" name="team" required>
                        {% for team in teams %}
                        <option value="{{ team }}">{{ team }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Starting Grid Position:</label>
                    <input type="number" id="grid" name="grid" min="1" max="20" value="5" required>
                </div>
                <button type="submit">Predict Finish Position</button>
            </form>
            <div id="raceResult"></div>
        </div>
        
        <div class="section">
            <h2>WDC Championship Simulator</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Run Monte Carlo simulations based on remaining races
            </p>
            <form id="wdcForm">
                <div class="form-group">
                    <label>Select Driver:</label>
                    <select id="wdcDriver" name="wdcDriver" required>
                        {% for driver in drivers %}
                        <option value="{{ driver }}" {% if driver == 'NOR' %}selected{% endif %}>{{ driver }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Number of Simulations:</label>
                    <input type="number" id="simulations" name="simulations" value="1000000" min="10000" max="10000000" step="10000" required>
                </div>
                <button type="submit" id="wdcButton">Run Championship Simulation</button>
            </form>
            <div class="loading" id="loading">
                ⏳ Running simulations... This may take a moment...
            </div>
            <div id="wdcResult"></div>
        </div>
    </div>
    
    <script>
        // Predict specific race winner
        async function predictSpecificRace(raceName) {
            console.log('Predicting race:', raceName);
            
            // Update active tab
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
                if (tab.textContent.includes(raceName.split(' ')[0])) {
                    tab.classList.add('active');
                }
            });
            
            document.getElementById('raceLoading').classList.add('active');
            document.getElementById('raceWinnerResult').innerHTML = '';
            
            try {
                const response = await fetch('/predict_race_winner', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ race_name: raceName })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                console.log('Received result:', result);
                
                document.getElementById('raceLoading').classList.remove('active');
                
                // Check if we have predictions
                if (!result.predictions || result.predictions.length === 0) {
                    document.getElementById('raceWinnerResult').innerHTML = `
                        <div class="error">
                            No predictions available. Check console for details.
                        </div>
                    `;
                    return;
                }
                
                const topDrivers = result.predictions.slice(0, 10);
                
                let html = `
                    <div class="result">
                        <h3>🏆 ${result.race_name} - Predicted Results</h3>
                        <ul class="driver-list">
                `;
                
                topDrivers.forEach((pred, idx) => {
                    const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `P${pred.predicted_position}`;
                    html += `
                        <li>
                            <span><strong>${medal}</strong> ${pred.driver} (${pred.team})</span>
                            <span>${pred.predicted_points} pts | ${pred.win_probability}% win</span>
                        </li>
                    `;
                });
                
                html += `
                        </ul>
                        <div class="success" style="margin-top: 20px;">
                            <strong>🏆 Predicted Winner:</strong> ${result.winner.driver} from ${result.winner.team}<br>
                            <strong>Win Probability:</strong> ${result.winner.win_probability}%
                        </div>
                    </div>
                `;
                
                document.getElementById('raceWinnerResult').innerHTML = html;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('raceLoading').classList.remove('active');
                document.getElementById('raceWinnerResult').innerHTML = `
                    <div class="error">
                        Error loading predictions: ${error.message}<br>
                        <small>Check browser console for more details</small>
                    </div>
                `;
            }
        }
        
        // Automatically predict Las Vegas on load
        window.onload = () => {
            setTimeout(() => {
                predictSpecificRace('Las Vegas Grand Prix');
            }, 500);
        };
        
        // Individual driver prediction
        document.getElementById('raceForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                // Get race emoji
                const raceEmojis = {
                    'Las Vegas Grand Prix': '🎰',
                    'Qatar Grand Prix': '🏜️',
                    'Abu Dhabi Grand Prix': '🏝️'
                };
                const emoji = raceEmojis[data.race] || '🏁';
                
                document.getElementById('raceResult').innerHTML = `
                    <div class="result">
                        <h3>Prediction Result - ${emoji} ${data.race}</h3>
                        <p><strong>${data.driver}</strong> (${data.team})</p>
                        <p>Starting from <strong>P${data.grid}</strong></p>
                        <div class="success" style="margin-top: 15px; font-size: 24px;">
                            Predicted Finish: <strong>P${result.predicted_position}</strong>
                        </div>
                        <p style="margin-top: 15px; color: #666; font-size: 14px;">
                            <strong>Points Expected:</strong> ${getPointsForPosition(result.predicted_position)} pts
                        </p>
                    </div>
                `;
            } catch (error) {
                document.getElementById('raceResult').innerHTML = `
                    <div class="error">
                        Error: ${error.message}
                    </div>
                `;
            }
        });
        
        // Helper function to calculate points
        function getPointsForPosition(position) {
            const points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1};
            return points[position] || 0;
        }
        
        // WDC simulation
        document.getElementById('wdcForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('wdcButton').disabled = true;
            document.getElementById('wdcResult').innerHTML = '';
            
            try {
                const response = await fetch('/simulate_wdc', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                document.getElementById('loading').classList.remove('active');
                document.getElementById('wdcButton').disabled = false;
                
                document.getElementById('wdcResult').innerHTML = `
                    <div class="result">
                        <h3>🏆 World Championship Simulation Results</h3>
                        <p style="margin-bottom: 15px; color: #666;">
                            Based on <strong>${result.remaining_races.length} remaining race(s)</strong>: 
                            <em>${result.remaining_races.join(', ')}</em>
                        </p>
                        <div class="stats">
                            <div class="stat-card">
                                <div class="stat-label">Driver</div>
                                <div class="stat-value">${data.wdcDriver}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Simulations Run</div>
                                <div class="stat-value">${parseInt(data.simulations).toLocaleString()}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Championships Won</div>
                                <div class="stat-value">${result.wins.toLocaleString()}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">WDC Win Probability</div>
                                <div class="stat-value">${result.probability.toFixed(2)}%</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Podium Rate</div>
                                <div class="stat-value">${result.stats.podium_rate.toFixed(1)}%</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Top 5 Rate</div>
                                <div class="stat-value">${result.stats.top5_rate.toFixed(1)}%</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Avg Final Points</div>
                                <div class="stat-value">${Math.round(result.stats.avg_points)}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Current Points</div>
                                <div class="stat-value">${result.stats.current_points}</div>
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('wdcButton').disabled = false;
                document.getElementById('wdcResult').innerHTML = `
                    <div class="error">
                        Error: ${error.message}
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
'''