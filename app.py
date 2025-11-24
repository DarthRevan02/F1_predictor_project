from flask import Flask, render_template_string, request, jsonify
from data_loader import F1DataLoader
from data_preprocessing import F1DataPreprocessor
from model_training import F1RaceModel
from prediction_modules.prediction_master import MasterF1Predictor
from simulation import WDCSimulator
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

app = Flask(__name__)

master_predictor = None
simulator = None
preprocessor = None

# Cache directory for storing predictions
CACHE_DIR = Path("prediction_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_filename(race_name):
    """Generate cache filename for a race"""
    safe_name = race_name.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_name}.json"

def save_prediction_to_cache(race_name, prediction_data):
    """Save prediction to cache file"""
    try:
        cache_file = get_cache_filename(race_name)
        with open(cache_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        print(f"✓ Cached predictions for {race_name}")
        return True
    except Exception as e:
        print(f"⚠️  Could not cache predictions: {e}")
        return False

def load_prediction_from_cache(race_name):
    """Load prediction from cache file"""
    try:
        cache_file = get_cache_filename(race_name)
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded cached predictions for {race_name}")
            return data
        return None
    except Exception as e:
        print(f"⚠️  Could not load cached predictions: {e}")
        return None

def clear_prediction_cache(race_name):
    """Clear cache for a specific race"""
    try:
        cache_file = get_cache_filename(race_name)
        if cache_file.exists():
            cache_file.unlink()
            print(f"✓ Cleared cache for {race_name}")
            return True
        return False
    except Exception as e:
        print(f"⚠️  Could not clear cache: {e}")
        return False

def get_race_status(race_name, year=2025):
    """Check if a race has ended"""
    try:
        import fastf1
        from datetime import datetime, timezone
        
        schedule = fastf1.get_event_schedule(year)
        current_time = datetime.now(timezone.utc)
        
        for idx, event in schedule.iterrows():
            if event['EventName'] == race_name:
                if pd.notna(event['EventDate']):
                    event_date = event['EventDate']
                    
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=timezone.utc)
                    
                    # Race has ended if current time is after event date
                    has_ended = current_time > event_date
                    
                    return {
                        'has_ended': has_ended,
                        'event_date': event_date.isoformat(),
                        'race_name': race_name
                    }
        
        # Race not found in schedule, assume not ended
        return {'has_ended': False, 'race_name': race_name}
    except Exception as e:
        app.logger.warning(f"Could not check race status: {e}")
        return {'has_ended': False, 'race_name': race_name}

def initialize_system():
    global master_predictor, simulator, preprocessor

    print("=" * 50)
    print("🏎️  F1 PREDICTION SYSTEM INITIALIZING")
    print("=" * 50)

    print("\n[1/4] Loading F1 race data...")
    print("   Loading 2024 and 2025 season data for training...")
    
    # Load data from both 2024 and 2025
    all_data = []
    years_to_load = [2024, 2025]
    
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
        'BOR': 19,    # Nico Borghesi :contentReference[oaicite:19]{index=19}
    }
    
    print(f"\n📊 Current Championship Standings (2025):")
    print(f"   P1: NOR - 390 pts (McLaren)")
    print(f"   P2: PIA - 366 pts (McLaren)")
    print(f"   P3: VER - 341 pts (Red Bull)")

    master_predictor = MasterF1Predictor(model, label_encoders, current_standings)
    simulator = WDCSimulator(preprocessor.get_drivers(), current_standings)
    
    # Connect qualifying predictor to simulator for grid predictions
    simulator.set_qualifying_predictor(master_predictor.qualifying_predictor)
    print("   ✓ Connected qualifying predictor to simulator")
    
    # Check existing cache
    cache_files = list(CACHE_DIR.glob("*.json"))
    if cache_files:
        print(f"\n📦 Found {len(cache_files)} cached prediction(s)")
    
    print("\n" + "=" * 50)
    print("✓ SYSTEM READY!")
    print(f"✓ Model trained on {len(all_data)} season(s): {', '.join(map(str, years_to_load))}")
    print(f"✓ Total training data: {len(df)} race results")
    print(f"✓ Predictions cache directory: {CACHE_DIR}")
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
    """Predict race winner with caching support"""
    data = request.get_json(silent=True) or {}
    race_name = data.get("race_name", "Next Race")
    force_refresh = data.get("force_refresh", False)

    try:
        # Check if race has ended
        race_status = get_race_status(race_name)
        
        # If trying to force refresh after race ended, deny it
        if force_refresh and race_status['has_ended']:
            app.logger.info(f"Denied refresh for {race_name} - race has ended")
            return jsonify({
                'error': 'Race has already ended',
                'message': f'{race_name} has already taken place. Predictions are locked.',
                'race_ended': True,
                'race_name': race_name
            }), 403
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_prediction = load_prediction_from_cache(race_name)
            if cached_prediction:
                app.logger.info(f"Returning cached prediction for {race_name}")
                cached_prediction['cached'] = True
                cached_prediction['race_ended'] = race_status['has_ended']
                return jsonify(cached_prediction)
        
        # Generate new prediction
        app.logger.info(f"Generating new prediction for {race_name}")
        raw_prediction = master_predictor.race_winner_predictor.predict_race_winner(race_name)
        
        # Clean numpy types
        cleaned = clean_numpy(raw_prediction)
        cleaned['cached'] = False
        cleaned['race_ended'] = race_status['has_ended']
        
        # Save to cache
        save_prediction_to_cache(race_name, cleaned)
        
        app.logger.info(f"Race winner prediction for {race_name}: {len(cleaned.get('predictions', []))} drivers")
        
        return jsonify(cleaned)
        
    except Exception as e:
        app.logger.error(f"Error in predict_race_winner: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'race_name': race_name,
            'predictions': [],
            'winner': {},
            'cached': False
        }), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear cache for a specific race"""
    data = request.get_json(silent=True) or {}
    race_name = data.get("race_name")
    
    if not race_name:
        return jsonify({'success': False, 'message': 'No race name provided'}), 400
    
    success = clear_prediction_cache(race_name)
    
    return jsonify({
        'success': success,
        'message': f'Cache cleared for {race_name}' if success else 'Cache not found'
    })


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
