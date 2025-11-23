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
    print("   Loading multiple seasons for better predictions...")
    
    # Load data from multiple years
    all_data = []
    years_to_load = [2024, 2025]  # Train on both 2024 and 2025
    
    for year in years_to_load:
        try:
            print(f"   → Loading {year} season data...")
            loader = F1DataLoader(year=year)
            df_year = loader.load_race_data(load_all=True)
            
            if not df_year.empty:
                all_data.append(df_year)
                print(f"   ✓ Loaded {len(df_year)} results from {year}")
            else:
                print(f"   ⚠️  No data found for {year} (season may not have started)")
        except Exception as e:
            print(f"   ⚠️  Error loading {year}: {e}")
    
    # Combine all data
    if not all_data:
        raise ValueError("❌ No F1 data available from any year. Please check your internet connection.")
    
    import pandas as pd
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(df)} total race results from {len(all_data)} season(s)")

    print("\n[2/4] Preprocessing data...")
    preprocessor = F1DataPreprocessor()
    df, label_encoders = preprocessor.preprocess_data(df)

    print("\n[3/4] Training machine learning model...")
    X = df[['Driver_Encoded', 'Team_Encoded', 'GridPosition']]
    y = df['Position']
    
    if len(X) == 0:
        raise ValueError("❌ No training data available after preprocessing!")
    
    print(f"   Training on {len(X)} race results...")
    model = F1RaceModel()
    model.train(X, y)

    print("\n[4/4] Initializing predictor and simulator...")
    # 2025 Championship Standings (Current)
    current_standings = {
        'NOR': 390, 'PIA': 366, 'VER': 341, 'RUS': 276, 'LEC': 214,
        'HAM': 148, 'ANT': 122, 'ALB': 73, 'HUL': 43, 'HAD': 43,
        'SAI': 30, 'LAW': 25, 'ALO': 22, 'GAS': 18, 'STR': 15,
        'OCO': 12, 'TSU': 10, 'BEA': 8, 'DOR': 5, 'BOR': 3
    }
    
    print(f"\n📊 Current Championship Standings (2025):")
    print(f"   P1: NOR - 390 pts (McLaren)")
    print(f"   P2: PIA - 366 pts (McLaren)")
    print(f"   P3: VER - 341 pts (Red Bull)")

    master_predictor = MasterF1Predictor(model, label_encoders, current_standings)
    simulator = WDCSimulator(preprocessor.get_drivers(), current_standings)
    
    print("\n" + "=" * 50)
    print("✓ SYSTEM READY!")
    print(f"✓ Model trained on data from: {', '.join(map(str, [y for y in years_to_load]))}")
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
        remaining_races = simulator.get_remaining_races_info(2024)
        if not remaining_races:
            # Fallback for 2024 end of season
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
            remaining_races = simulator.get_remaining_races_info(2024)
            if not remaining_races:
                remaining_races = ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']
        except:
            remaining_races = ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']
        
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