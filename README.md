# 🏎️ F1 Race Predictor & WDC Simulator

A comprehensive Formula 1 prediction system powered by **Machine Learning** and **Monte Carlo simulations**. Predict race winners, championship probabilities, and analyze every aspect of F1 racing with 10 specialized prediction modules.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastF1](https://img.shields.io/badge/FastF1-3.3.5-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

### 🏁 **Race Predictions**
- **Race Winner Prediction** — ML-powered top 10 predictions
- **Podium Analysis** — Monte Carlo probability calculations
- **Qualifying Simulation** — Starting grid predictions
- **Fastest Lap Prediction** — Pace and strategy analysis

### 🏆 **Championship Analysis**
- **WDC Simulator** — 1 million Monte Carlo simulations
- **Championship Impact** — Race-by-race standings effects
- **Title Probability** — Real-time championship odds

### 📊 **Advanced Analytics**
- **Points Finish Predictor** — Top 10 scoring probabilities
- **DNF Risk Analysis** — Reliability and driver risk factors
- **Overtaking Predictions** — Position changes and action
- **Team Performance** — Constructor championship analysis
- **Strategy Predictor** — Pit stop and tire compound forecasting

### 🎯 **2025 Season Ready**
- ✅ Complete 2025 driver lineup
- ✅ Updated team performance ratings
- ✅ Final 2025 WDC standings (NOR: 410, VER: 408, PIA: 395)
- ✅ Real-time data from FastF1 API

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for F1 data)

### Installation

1. **Clone or download the repository**
```bash
git clone <your-repo-url>
cd f1_predictor_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open in browser**
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
f1_predictor_project/
├── 📄 requirements.txt               # Python dependencies (5 packages)
├── 📄 README.md                      # This file
├── 📄 __init__.py                    # Root package init
│
├── 🔧 Core System
│   ├── app.py                        # Flask app — 7 routes, auth, cache, async sim jobs
│   ├── data_loader.py                # FastF1 data fetching (2024 & 2025 seasons)
│   ├── data_preprocessing.py         # Label encoding, driver/team normalisation
│   ├── model_training.py             # Linear Regression training & prediction
│   ├── simulation.py                 # Monte Carlo WDC simulator (1M+ sims, async-safe)
│   ├── driver_config.py              # ⭐ Single source of truth: standings, ratings, mappings
│   └── templates.py                  # Inline Flask HTML template (no /templates dir)
│
├── 🎯 prediction_modules/
│   ├── __init__.py                   # Package init
│   ├── prediction_master.py          # Master coordinator — runs all 10 modules
│   ├── prediction_race_winner.py     # ML-based top-10 race predictor
│   ├── prediction_podium.py          # Monte Carlo podium probabilities (1,000 sims)
│   ├── prediction_qualifying.py      # Grid position predictor (quali skill × team strength)
│   ├── prediction_fastest_lap.py     # Fastest lap setter predictor (pace + strategy)
│   ├── prediction_points_finish.py   # Points-finish probability (1,000 sims)
│   ├── prediction_dnf_probability.py # DNF risk: reliability + driver + track factors
│   ├── prediction_overtakes.py       # Position-change & overtaking statistics
│   ├── prediction_championship_impact.py  # Race-by-race WDC standing shifts
│   ├── prediction_team_performance.py     # Constructor-level points & podium predictions
│   └── prediction_strategy.py             # Pit stop count, lap windows & tyre compounds
│
└── 📂 Auto-generated Cache
    ├── f1_cache/                     # FastF1 session data (created on first run)
    └── prediction_cache/             # Per-race JSON snapshots
        ├── Las_Vegas_Grand_Prix.json # Cached Las Vegas prediction
        └── Qatar_Grand_Prix.json     # Cached Qatar prediction
```

> **`driver_config.py` is the single configuration file** for all driver/team data. Editing it propagates changes automatically to every prediction module — no need to touch individual module files.

---

## 🎮 Usage

### Web Interface

#### **1. Race Predictions**
Click on race tabs to see predictions:
- 🎰 **Las Vegas GP**
- 🏜️ **Qatar GP**
- 🏝️ **Abu Dhabi GP**

Each shows:
- Predicted winner
- Top 10 finishers with points
- Podium finishers (highlighted)

#### **2. Individual Driver Prediction**
- Select driver and team
- Enter starting grid position
- Get predicted finish position

#### **3. WDC Championship Simulator**
- Select driver
- Choose simulation count (default: 1,000,000)
- Get championship probability with detailed stats

---

## 📊 Prediction Modules Explained

### **1. Race Winner Predictor**
Predicts race winner and top 10 using ML model trained on historical data.

**Output:**
- Predicted winner with win probability
- Top 10 finishing order
- Expected championship points

### **2. Podium Predictor**
Runs 1,000 race simulations to calculate podium probabilities.

**Output:**
- P1, P2, P3 individual probabilities
- Overall podium probability
- Top 6 contenders

### **3. Qualifying Predictor**
Predicts starting grid based on team/driver qualifying strength.

**Output:**
- Complete predicted grid (P1–P20)
- Pole position favorite
- Q3 advancement probabilities

### **4. Fastest Lap Predictor**
Analyzes pace and strategy to predict fastest lap setter.

**Output:**
- Fastest lap probability per driver
- Pace scores
- Pit stop strategy likelihood

### **5. Points Finish Predictor**
Simulates races to predict top 10 points scorers.

**Output:**
- Points scoring probability
- Average finishing position
- Best/worst case scenarios

### **6. DNF Probability Predictor**
Analyzes reliability and driver risk for retirement predictions.

**Output:**
- DNF probability per driver
- Reliability risk factors
- Driver mistake risk
- Finish probability

### **7. Overtake Predictor**
Predicts position changes and overtaking action.

**Output:**
- Expected overtakes made
- Positions lost
- Net position change
- Overtaking/defensive ratings

### **8. Championship Impact Predictor**
Shows how race results affect championship standings.

**Output:**
- Points after race (avg/best/worst)
- Championship position changes
- Title probability changes

### **9. Team Performance Predictor**
Predicts team-level metrics for constructors' championship.

**Output:**
- Expected team points (both cars)
- Best driver position
- Both cars in points probability
- Podium probability

### **10. Strategy Predictor**
Forecasts pit stop strategies and tire choices.

**Output:**
- Strategy type (1-stop, 2-stop, etc.)
- Pit stop lap windows
- Tire compound choices
- Strategy risk levels
- Undercut probabilities

---

## 🔬 Technology Stack

### **Machine Learning**
- **scikit-learn** — Linear Regression model
- **NumPy** — Numerical computations
- **Pandas** — Data manipulation

### **Data Source**
- **FastF1** — Official F1 data API
- Real telemetry and race results
- Historical performance data

### **Web Framework**
- **Flask** — Python web framework
- RESTful API endpoints
- Real-time predictions

### **Simulation**
- **Monte Carlo Method** — 1M+ simulations
- Probabilistic modeling
- Statistical analysis

---

## 🎯 2025 Season Data

### **Final 2025 Championship Standings**

| Pos | Driver | Team | Points | Wins |
|-----|--------|------|--------|------|
| 🏆 1 | L. Norris 🇬🇧 | McLaren | 410 | 7 |
| 2 | M. Verstappen 🇳🇱 | Red Bull | 408 | 8 |
| 3 | O. Piastri 🇦🇺 | McLaren | 395 | 7 |
| 4 | G. Russell 🇬🇧 | Mercedes | 276 | 2 |
| 5 | C. Leclerc 🇲🇨 | Ferrari | 214 | 0 |
| 6 | C. Sainz 🇪🇸 | Williams | 163 | 0 |
| 7 | L. Hamilton 🇬🇧 | Ferrari | 152 | 0 |
| 8 | A.K. Antonelli 🇮🇹 | Mercedes | 137 | 0 |
| 9 | A. Albon 🇹🇭 | Williams | 73 | 0 |
| 10 | I. Hadjar 🇫🇷 | Racing Bulls | 54 | 0 |

> **Note:** Lando Norris won the 2025 World Drivers' Championship by just **2 points** over Verstappen — the closest title fight since 2021.

### **2025 Driver Lineup**

- **Ferrari**: Hamilton, Leclerc
- **Red Bull Racing**: Verstappen, Tsunoda *(Lawson demoted to Racing Bulls mid-season)*
- **McLaren**: Norris, Piastri
- **Mercedes**: Russell, Antonelli
- **Aston Martin**: Alonso, Stroll
- **Alpine**: Gasly, Colapinto *(replaced Doohan mid-season)*
- **Haas**: Ocon, Bearman
- **Racing Bulls**: Lawson, Hadjar
- **Williams**: Sainz, Albon
- **Kick Sauber**: Hülkenberg, Bortoleto

---

## 🏁 Predicted vs Actual Results — 2025 Final 3 Races

### 🎰 Las Vegas Grand Prix

| Pos | Predicted | Actual |
|-----|-----------|--------|
| P1 | **VER** ✅ | VER |
| P2 | PIA ❌ | RUS |
| P3 | NOR ❌ | ANT |
| P4 | LEC ✅ | LEC |
| P5 | LAW ❌ | SAI |
| P6 | RUS ❌ | HAD |
| P7 | HAM ❌ | HUL |
| P8 | ANT ❌ | HAM |
| P9 | ALO ❌ | OCO |
| P10 | STR ❌ | BEA |

> ⚠️ NOR and PIA were **disqualified** post-race for excessive skid wear, significantly reshuffling positions P2–P10.

### 🏜️ Qatar Grand Prix

| Pos | Predicted | Actual |
|-----|-----------|--------|
| P1 | **VER** ✅ | VER |
| P2 | NOR ❌ | PIA |
| P3 | LAW ❌ | SAI |
| P4 | PIA ❌ | NOR |
| P5 | LEC ❌ | ANT |
| P6 | HAM ❌ | RUS |
| P7 | RUS ❌ | ALO |
| P8 | STR ❌ | LEC |
| P9 | ALO ❌ | LAW ✅ |
| P10 | ANT ❌ | TSU |

### 🏝️ Abu Dhabi Grand Prix

| Pos | Predicted | Actual |
|-----|-----------|--------|
| P1 | **VER** ✅ | VER |
| P2 | **PIA** ✅ | PIA |
| P3 | LAW ❌ | NOR |
| P4 | HAM ❌ | LEC |
| P5 | LEC ❌ | RUS |
| P6 | NOR ❌ | ALO |
| P7 | ANT ❌ | OCO |
| P8 | STR ❌ | HAM |
| P9 | RUS ❌ | HUL |
| P10 | OCO ❌ | STR |

---

## 📈 Prediction Accuracy Analysis

### Per-Race Metrics

| Metric | Las Vegas | Qatar | Abu Dhabi |
|--------|-----------|-------|-----------|
| Winner correct | ✅ Yes | ✅ Yes | ✅ Yes |
| Podium overlap | 1 / 3 | 1 / 3 | 2 / 3 |
| Top-10 drivers overlap | 5 / 10 | 8 / 10 | 8 / 10 |
| Exact position matches | 2 / 5* | 1 / 8* | 2 / 8* |
| MAE (positions) | 2.00 | 2.62 | 2.12 |
| R² (prediction vs actual) | −0.438 | −0.495 | 0.191 |

*\* Calculated only over drivers appearing in both predicted and actual top 10.*

### Overall Accuracy (All 3 Races Combined)

| Metric | Value |
|--------|-------|
| **Winner accuracy** | **3 / 3 (100%)** |
| Average MAE | **2.29 positions** |
| **Overall R²** | **−0.112** |
| **Training R²** | **0.480** |

### Interpreting the Results

**Winner prediction was perfect** — VER was correctly predicted to win all three races, which he did.

**Overall R² of −0.112** means the model's position-by-position rankings performed below a simple baseline (predicting every driver finishes at the mean position) across the full top 10. This is largely explained by two factors:

1. **The McLaren double DSQ at Las Vegas** was an unpredictable post-race event that reshuffled P2–P10 entirely, making predictions look much worse than on-track performance warranted.
2. **Midfield volatility** — drivers like SAI (Williams), HAD, HUL, and COL repeatedly overperformed their team ratings, while the model over-ranked established names (LAW, STR, ALO) in the lower points positions.

**Training R² of 0.480** reflects the linear model's fit on historical 2024–2025 data. The gap between training R² and prediction R² indicates the model generalizes the top-end well but struggles to rank the competitive midfield.

---

## 🛠️ API Endpoints

These are the **7 routes implemented in `app.py`**. The prediction sub-modules (podium, qualifying, DNF, etc.) are called internally by `/predict_race_winner` and the master predictor — they are not exposed as separate HTTP endpoints.

### **Page**

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Web UI — race tabs, driver predictor, WDC simulator |

### **Race Prediction**

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/predict_race_winner` | Top-10 prediction for a named race. Returns cached result on repeat calls; pass `force_refresh: true` to regenerate. |
| `POST` | `/predict` | Single-driver finish position from driver, team, and grid inputs. |

### **Cache Management**

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/save_prediction` | Explicitly persist the current in-memory prediction to the server-side JSON cache. |
| `POST` | `/clear_cache` | Delete the cached prediction for a named race so the next fetch regenerates it. |

### **WDC Simulation (Async)**

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/simulate_wdc` | Start a background Monte Carlo simulation. Returns a `job_id` immediately — the page stays responsive. |
| `GET` | `/sim_status/<job_id>` | Poll for simulation results. Returns `{status: "running"}` until complete, then full stats. |

### **Request / Response Examples**

```python
import requests, time

# ── Race winner prediction ────────────────────────────────────────
r = requests.post('http://127.0.0.1:5000/predict_race_winner',
                  json={'race_name': 'Abu Dhabi Grand Prix'})
data = r.json()
print(data['winner']['driver'], data['winner']['win_probability'])
# → VER  78.9

# Force a fresh prediction (ignores cache)
r = requests.post('http://127.0.0.1:5000/predict_race_winner',
                  json={'race_name': 'Abu Dhabi Grand Prix', 'force_refresh': True})

# ── Single driver prediction ──────────────────────────────────────
r = requests.post('http://127.0.0.1:5000/predict',
                  json={'driver': 'NOR', 'team': 'McLaren', 'grid': 3})
print(r.json()['predicted_position'])   # → 2

# ── Save current prediction to cache ─────────────────────────────
requests.post('http://127.0.0.1:5000/save_prediction',
              json={'race_name': 'Abu Dhabi Grand Prix', 'prediction': data})

# ── Clear cache for a race ────────────────────────────────────────
requests.post('http://127.0.0.1:5000/clear_cache',
              json={'race_name': 'Abu Dhabi Grand Prix'})

# ── WDC simulation (async) ────────────────────────────────────────
job = requests.post('http://127.0.0.1:5000/simulate_wdc',
                    json={'wdcDriver': 'NOR', 'simulations': 1_000_000,
                          'seed': 42}).json()

while True:
    status = requests.get(f"http://127.0.0.1:5000/sim_status/{job['job_id']}").json()
    if status['status'] == 'done':
        print(f"{status['probability']:.2f}% WDC probability")
        print(f"Podium rate: {status['stats']['podium_rate']:.1f}%")
        break
    time.sleep(2)
```

### **Optional Authentication**

Set the `API_KEY` environment variable to protect all mutating endpoints. If set, every `POST` request must include the header `X-API-Key: <your-key>`. Leave unset to disable auth (development mode).

```bash
export API_KEY=my-secret-key
python app.py
```

---

## ⚙️ Configuration

### **Update Championship Standings**

Edit `driver_config.py` (single source of truth — one edit updates all modules):

```python
CURRENT_STANDINGS = {
    'NOR': 410,  # 2025 World Champion
    'VER': 408,
    'PIA': 395,
    # ... etc
}
```

### **Change Simulation Count**

```python
# In web interface or API calls
num_simulations = 1_000_000  # High accuracy, slower
num_simulations = 100_000    # Good accuracy, faster
```

### **Modify Driver Performance**

Edit `driver_config.py` to adjust ratings (0.0–1.0). Changes propagate to all 10 prediction modules automatically:

```python
DRIVER_PERFORMANCE = {
    'NOR': 0.95,
    'PIA': 0.93,
    # ... etc
}
```

---

## 🧪 Testing

```bash
# Check if packages are installed
python -c "import fastf1, flask, pandas, sklearn; print('✓ All packages OK')"

# Test data loader
python -c "from data_loader import F1DataLoader; print('✓ Data loader OK')"

# Test predictions
python -c "from prediction_modules.prediction_master import MasterF1Predictor; print('✓ Predictions OK')"
```

---

## 🐛 Troubleshooting

### **ModuleNotFoundError**
```bash
pip install fastf1 flask pandas numpy scikit-learn
```

### **Port already in use**
```python
# Change port in app.py
app.run(debug=True, port=5001)
```

### **FastF1 data errors**
```bash
rm -rf f1_cache/   # Mac/Linux
python app.py
```

### **Slow simulations**
```python
num_simulations = 100_000  # Reduce from 1,000,000
```

---

## 📈 Performance

### **Model Metrics**

| Metric | Value |
|--------|-------|
| Training R² (Linear Regression) | 0.480 |
| Training samples | 958 results |
| Prediction R² (2025 final 3 races) | −0.112 |
| Winner accuracy | 100% (3/3) |
| Average position MAE | 2.29 |

### **Speed**

- **First Run**: 5–10 minutes (downloads FastF1 data)
- **Subsequent Runs**: 30–60 seconds (uses cache)
- **1M Simulations**: 20–60 seconds (background thread)
- **Race Prediction**: < 1 second

### **System Requirements**

- **Minimum**: Python 3.8, 2 GB RAM, 500 MB disk
- **Recommended**: Python 3.10+, 4 GB RAM, 1 GB disk

---

## 🤝 Contributing

Areas for improvement:

1. **Better ML models** — Random Forest or XGBoost for non-linear relationships
2. **Weather integration** — Rain/safety car probability
3. **Live race predictions** — Real-time lap-by-lap updates
4. **Disqualification/DNF modeling** — Post-race event handling
5. **Midfield-specific ratings** — Separate tiers for top teams vs midfield

---

## 📝 License

MIT License — feel free to use and modify!

---

## 🙏 Acknowledgments

- **FastF1** — For providing excellent F1 data API
- **scikit-learn** — For machine learning tools
- **Flask** — For web framework
- **F1 Community** — For inspiration and feedback

---

## 🔮 Roadmap

### **v1.0** (Current)
- ✅ 10 prediction modules
- ✅ Async WDC simulator
- ✅ 2025 season data
- ✅ Web interface with cache & save

### **v1.1** (Planned)
- [ ] Weather / safety car integration
- [ ] Random Forest / XGBoost model upgrade
- [ ] Driver comparison tool
- [ ] Export to PDF/CSV

### **v2.0** (Future)
- [ ] Neural network models
- [ ] Live race predictions
- [ ] Mobile app
- [ ] Fantasy F1 integration

---

## 📸 Interface Preview

```
┌─────────────────────────────────────────────┐
│  🏁 Formula 1 Prediction System              │
├─────────────────────────────────────────────┤
│  🎰 Las Vegas │ 🏜️ Qatar │ 🏝️ Abu Dhabi      │
│  🔄 Generate New   💾 Save Prediction        │
├─────────────────────────────────────────────┤
│  🥇 VER (Red Bull Racing) — 25 pts | 78.9%  │
│  🥈 PIA (McLaren)         — 18 pts | 40%    │
│  🥉 NOR (McLaren)         — 15 pts | 32%    │
│     LEC (Ferrari)         — 12 pts | 24%    │
│     ...                                     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  🏆 Championship Simulation Results          │
├─────────────────────────────────────────────┤
│  Driver:          Lando Norris               │
│  Simulations:     1,000,000                  │
│  WDC Probability: 95.23%                     │
│  Podium Rate:     92.4%                      │
│  Avg Final Pts:   438                        │
└─────────────────────────────────────────────┘
```

---

**Built with ❤️ for F1 fans by F1 fans**

*Last Updated: 2025 Season Complete — Lando Norris, World Champion* 🏆
