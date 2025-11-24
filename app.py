from flask import Flask, render_template_string, request, jsonify
from data_loader import F1DataLoader
from data_preprocessing import F1DataPreprocessor
from model_training import F1RaceModel
from prediction_modules.prediction_master import MasterF1Predictor
from simulation import WDCSimulator
import numpy as np

app = Flask(__name__)

master_predictor = None
simulator = None
preprocessor = None

def initialize_system():
    global master_predictor, simulator, preprocessor

    print("=" * 50)
    print("🏎️  F1 PREDICTION SYSTEM INITIALIZING")
    print("=" * 50)

    print("\n[1/4] Loading F1 race data...")
    print("   Loading 2025 season data only...")
    
    # Load data from 2025 ONLY
    try:
        print(f"   → Loading 2025 season data...")
        loader = F1DataLoader(year=2025)
        df = loader.load_race_data(load_all=True)
        
        if df.empty:
            raise ValueError("❌ No data found for 2025 season. The season may not have started yet or no races have been completed.")
        
        print(f"\n✓ Loaded dataset: {len(df)} total race results from 2025 season")
        
    except Exception as e:
        print(f"   ⚠️  Error loading 2025 data: {e}")
        raise ValueError("❌ No F1 data available from 2025. Please check your internet connection or wait for the season to start.")

    print("\n[2/4] Preprocessing data...")
    preprocessor = F1DataPreprocessor()
    df, label_encoders = preprocessor.preprocess_data(df)

    print("\n[3/4] Training machine learning model...")
    X = df[['Driver_Encoded', 'Team_Encoded', 'GridPosition']]
    y = df['Position']
    
    if len(X) == 0:
        raise ValueError("❌ No training data available after preprocessing!")
    
    print(f"   Training on {len(X)} race results from 2025...")
    model = F1RaceModel()
    model.train(X, y)

    print("\n[4/4] Initializing predictor and simulator...")
    # 2025 Championship Standings (Current)
    current_standings = {
        'NOR': 390,   # Lando Norris :contentReference[oaicite:1]{index=1}
        'PIA': 366,   # Oscar Piastri :contentReference[oaicite:2]{index=2}
        'VER': 366,   # Max Verstappen :contentReference[oaicite:3]{index=3}
        'RUS': 294,   # George Russell :contentReference[oaicite:4]{index=4}
        'LEC': 226,   # Charles Leclerc :contentReference[oaicite:5]{index=5}
        'HAM': 152,   # Lewis Hamilton :contentReference[oaicite:6]{index=6}
        'ANT': 137,   # Kimi Antonelli :contentReference[oaicite:7]{index=7}
        'ALB': 73,    # Alexander Albon :contentReference[oaicite:8]{index=8}
        'HAD': 51,    # Isack Hadjar :contentReference[oaicite:9]{index=9}
        'HUL': 49,    # Nico Hülkenberg :contentReference[oaicite:10]{index=10}
        'SAI': 48,    # Carlos Sainz :contentReference[oaicite:11]{index=11}
        'BEA': 41,    # Oliver Bearman :contentReference[oaicite:12]{index=12}
        'ALO': 40,    # Fernando Alonso :contentReference[oaicite:13]{index=13}
        'LAW': 36,    # Liam Lawson :contentReference[oaicite:14]{index=14}
        'OCO': 32,    # Esteban Ocon :contentReference[oaicite:15]{index=15}
        'STR': 32,    # Lance Stroll :contentReference[oaicite:16]{index=16}
        'TSU': 28,    # Yuki Tsunoda :contentReference[oaicite:17]{index=17}
        'GAS': 22,    # Pierre Gasly :contentReference[oaicite:18]{index=18}
        'BOR': 19
    }
    
    print(f"\n📊 Current Championship Standings (2025):")
    print(f"   P1: NOR - 390 pts (McLaren)")
    print(f"   P2: PIA - 366 pts (McLaren)")
    print(f"   P3: VER - 341 pts (Red Bull)")

    master_predictor = MasterF1Predictor(model, label_encoders, current_standings)
    simulator = WDCSimulator(preprocessor.get_drivers(), current_standings)
    
    print("\n" + "=" * 50)
    print("✓ SYSTEM READY!")
    print(f"✓ Model trained on 2025 season data only")
    print("=" * 50)
    print("\n🌐 Open http://127.0.0.1:5000 in your browser\n")


def clean_numpy(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy(i) for i in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------- ROUTES ----------------------

@app.route('/')
def home():
    from templates import get_html_template
    drivers = preprocessor.get_drivers()
    teams = preprocessor.get_teams()
    
    # Get remaining races - fallback to manual list if API fails
    try:
        remaining_races = simulator.get_remaining_races_info(2025)
        if not remaining_races:
            # Fallback for 2025 end of season
            remaining_races = ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']
    except Exception as e:
        app.logger.warning(f"Could not fetch remaining races: {e}")
        remaining_races = ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']

    return render_template_string(
        get_html_template(),
        drivers=drivers,
        teams=teams,
        remaining_races=remaining_races
    )


@app.route('/predict_race_winner', methods=['POST'])
def predict_race_winner():
    """Predict race winner - FIXED VERSION"""
    data = request.get_json(silent=True) or {}
    race_name = data.get("race_name", "Next Race")

    try:
        # Call the race_winner_predictor directly (not get_comprehensive_predictions)
        raw_prediction = master_predictor.race_winner_predictor.predict_race_winner(race_name)
        
        # Clean numpy types
        cleaned = clean_numpy(raw_prediction)
        
        app.logger.info(f"Race winner prediction for {race_name}: {len(cleaned.get('predictions', []))} drivers")
        
        return jsonify(cleaned)
        
    except Exception as e:
        app.logger.error(f"Error in predict_race_winner: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'race_name': race_name,
            'predictions': [],
            'winner': {}
        }), 500


@app.route('/predict', methods=['POST'])
def predict_single_driver():
    """Individual driver position prediction"""
    data = request.json

    driver = data.get("driver")
    team = data.get("team")
    grid = int(data.get("grid"))

    try:
        # Encode driver and team
        driver_encoded = master_predictor.label_encoders['Driver'].transform([driver])[0]
        team_encoded = master_predictor.label_encoders['Team'].transform([team])[0]
        
        # Predict
        X_pred = np.array([[driver_encoded, team_encoded, grid]])
        predicted_pos = master_predictor.model.predict(X_pred)
        
        # Handle both array and scalar returns
        if isinstance(predicted_pos, (list, np.ndarray)):
            predicted_pos = predicted_pos[0]
        
        return jsonify({"predicted_position": int(round(max(1, predicted_pos)))})
        
    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/simulate_wdc', methods=['POST'])
def simulate_wdc():
    """WDC championship simulation"""
    data = request.json
    driver = data['wdcDriver']
    num_sims = int(data['simulations'])
    
    try:
        # Get remaining races with fallback
        try:
            remaining_races = simulator.get_remaining_races_info(2025)
            if not remaining_races:
                remaining_races = ['Qatar Grand Prix', 'Abu Dhabi Grand Prix']
        except:
            remaining_races = ['Qatar Grand Prix', 'Abu Dhabi Grand Prix']
        
        probability, wins, stats = simulator.simulate_season(driver, remaining_races, num_sims)
        
        result = {
            'probability': probability,
            'wins': wins,
            'stats': stats,
            'remaining_races': remaining_races
        }
        
        return jsonify(clean_numpy(result))
        
    except Exception as e:
        app.logger.error(f"Error in simulate_wdc: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# MAIN
if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=5000)