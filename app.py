import os
import json
import uuid
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

from data_loader import F1DataLoader
from data_preprocessing import F1DataPreprocessor
from model_training import F1RaceModel
from prediction_modules.prediction_master import MasterF1Predictor
from simulation import WDCSimulator
from driver_config import CURRENT_STANDINGS
from templates import get_html_template

app = Flask(__name__)

# ── Auth ──────────────────────────────────────────────────────────────────────
# Set API_KEY in the environment to protect every mutating endpoint.
# Example: export API_KEY=my-secret-key
# Leave unset to disable auth (development only — fixes flaw #2).
_API_KEY = os.environ.get('API_KEY', '')


def _check_auth():
    """Return a 401 response if auth is enabled and the header is wrong."""
    if not _API_KEY:
        return None   # auth disabled
    key = request.headers.get('X-API-Key', '')
    if key != _API_KEY:
        return jsonify({'error': 'Unauthorised'}), 401
    return None


# ── App state ─────────────────────────────────────────────────────────────────
# Wrapped in a single object so the reference is stable after init (flaw #6).
class _AppState:
    master_predictor: MasterF1Predictor = None
    simulator: WDCSimulator = None
    preprocessor: F1DataPreprocessor = None


_state = _AppState()

# ── Prediction cache ──────────────────────────────────────────────────────────
CACHE_DIR = Path('prediction_cache')
CACHE_DIR.mkdir(exist_ok=True)

# Per-race lock prevents two simultaneous requests from double-writing cache
# (fixes flaw #13).
_cache_locks: dict[str, threading.Lock] = {}
_cache_locks_lock = threading.Lock()   # protects the dict itself


def _get_race_lock(race_name: str) -> threading.Lock:
    with _cache_locks_lock:
        if race_name not in _cache_locks:
            _cache_locks[race_name] = threading.Lock()
        return _cache_locks[race_name]


def _cache_path(race_name: str) -> Path:
    safe = race_name.replace(' ', '_').replace('/', '_')
    return CACHE_DIR / f'{safe}.json'


def _load_cache(race_name: str):
    try:
        p = _cache_path(race_name)
        if p.exists():
            with p.open() as f:
                return json.load(f)
    except Exception as e:
        app.logger.warning(f'Cache load failed: {e}')
    return None


def _save_cache(race_name: str, data: dict):
    try:
        with _cache_path(race_name).open('w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        app.logger.warning(f'Cache save failed: {e}')


def _clear_cache(race_name: str):
    try:
        p = _cache_path(race_name)
        if p.exists():
            p.unlink()
            return True
    except Exception as e:
        app.logger.warning(f'Cache clear failed: {e}')
    return False


# ── Background simulation jobs ────────────────────────────────────────────────
# Simulations run in a worker thread so Flask stays responsive (flaw #5).
_sim_jobs: dict[str, dict] = {}
_sim_jobs_lock = threading.Lock()


def _run_sim(job_id: str, driver: str, num_sims: int, remaining_races: list, seed):
    try:
        prob, wins, stats = _state.simulator.simulate_season(
            driver, remaining_races, num_sims, seed=seed
        )
        result = {
            'status': 'done',
            'probability': prob,
            'wins': wins,
            'stats': stats,
            'remaining_races': remaining_races,
        }
    except Exception as e:
        result = {'status': 'error', 'error': str(e)}

    with _sim_jobs_lock:
        _sim_jobs[job_id].update(result)


# ── System initialisation ─────────────────────────────────────────────────────

def initialize_system():
    print('=' * 55)
    print('🏎️  F1 PREDICTION SYSTEM INITIALISING')
    print('=' * 55)

    all_data = []
    for year in [2024, 2025]:
        try:
            loader = F1DataLoader(year=year)
            df_year = loader.load_race_data(load_all=True)
            if not df_year.empty:
                all_data.append(df_year)
                print(f'✓ Loaded {len(df_year)} results from {year}')
            else:
                print(f'⚠️  No data for {year}')
        except Exception as e:
            print(f'⚠️  Error loading {year}: {e}')

    if not all_data:
        raise RuntimeError('No F1 data available from any year.')

    df = pd.concat(all_data, ignore_index=True)

    # Guard: model.train() raises ValueError on empty input (flaw #9 handled
    # inside F1RaceModel, but we also check here for a clear message).
    if df.empty:
        raise RuntimeError('Combined dataset is empty after loading — cannot train model.')

    print(f'\n✓ Combined dataset: {len(df)} results')

    _state.preprocessor = F1DataPreprocessor()
    df, label_encoders = _state.preprocessor.preprocess_data(df)

    X = df[['Driver_Encoded', 'Team_Encoded', 'GridPosition']]
    y = df['Position']
    model = F1RaceModel()
    model.train(X, y)

    _state.master_predictor = MasterF1Predictor(model, label_encoders, CURRENT_STANDINGS)
    _state.simulator = WDCSimulator(_state.preprocessor.get_drivers(), CURRENT_STANDINGS)
    _state.simulator.set_qualifying_predictor(_state.master_predictor.qualifying_predictor)

    cached = list(CACHE_DIR.glob('*.json'))
    if cached:
        print(f'📦 {len(cached)} cached prediction file(s) found')

    print('\n' + '=' * 55)
    print('✓ SYSTEM READY')
    print('=' * 55)
    print('\n🌐 Open http://127.0.0.1:5000\n')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(obj):
    """Recursively convert numpy types to native Python (for jsonify)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _get_race_status(race_name, year=2025):
    try:
        import fastf1
        from datetime import datetime, timezone
        schedule = fastf1.get_event_schedule(year)
        now = datetime.now(timezone.utc)
        for _, event in schedule.iterrows():
            if event['EventName'] == race_name:
                ed = event['EventDate']
                if ed.tzinfo is None:
                    ed = ed.replace(tzinfo=timezone.utc)
                return {'has_ended': now > ed, 'race_name': race_name}
    except Exception:
        pass
    return {'has_ended': False, 'race_name': race_name}


# ── Input validation helpers ──────────────────────────────────────────────────

def _validate_driver(driver: str):
    """Return error message if driver is not in the known set, else None."""
    if not driver or not isinstance(driver, str):
        return 'driver is required'
    known = set(_state.preprocessor.get_drivers())
    if driver not in known:
        return f'Unknown driver: {driver!r}'
    return None


def _validate_team(team: str):
    if not team or not isinstance(team, str):
        return 'team is required'
    known = set(_state.preprocessor.get_teams())
    if team not in known:
        return f'Unknown team: {team!r}'
    return None


def _validate_grid(raw):
    """Parse and range-check a grid position. Returns (int, error_string)."""
    try:
        grid = int(raw)
    except (TypeError, ValueError):
        return None, 'grid must be an integer'
    if not (1 <= grid <= 20):
        return None, 'grid must be between 1 and 20'
    return grid, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    drivers = _state.preprocessor.get_drivers()
    teams = _state.preprocessor.get_teams()
    try:
        remaining = _state.simulator.get_remaining_races_info(2025) or [
            'Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix'
        ]
    except Exception:
        remaining = ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']

    return render_template_string(
        get_html_template(),
        drivers=drivers,
        teams=teams,
        remaining_races=remaining,
    )


@app.route('/predict_race_winner', methods=['POST'])
def predict_race_winner():
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}
    race_name = str(data.get('race_name', 'Next Race'))
    force_refresh = bool(data.get('force_refresh', False))

    race_status = _get_race_status(race_name)

    # Race-ended no longer blocks regeneration — user can always re-predict
    # and choose whether to persist via the Save button (/save_prediction).
    # Auto-save only happens on the first fetch (not force_refresh) so that
    # a manually saved prediction is never silently overwritten.
    race_lock = _get_race_lock(race_name)
    with race_lock:
        if not force_refresh:
            cached = _load_cache(race_name)
            if cached:
                cached['cached'] = True
                cached['race_ended'] = race_status['has_ended']
                return jsonify(cached)

        raw = _state.master_predictor.race_winner_predictor.predict_race_winner(race_name)
        cleaned = _clean(raw)
        cleaned['cached'] = False
        cleaned['race_ended'] = race_status['has_ended']

        # Only auto-persist on first-time fetch, not on user-triggered re-predicts
        if not force_refresh:
            _save_cache(race_name, cleaned)

    return jsonify(cleaned)


@app.route('/clear_cache', methods=['POST'])
def clear_cache_route():
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}
    race_name = data.get('race_name')
    if not race_name:
        return jsonify({'success': False, 'message': 'race_name required'}), 400

    success = _clear_cache(str(race_name))
    return jsonify({'success': success})


@app.route('/save_prediction', methods=['POST'])
def save_prediction_route():
    """
    Explicitly save (or overwrite) the current prediction for a race to
    the server-side cache.  The client sends the full prediction payload
    it already has in memory — no re-computation required.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}
    race_name = data.get('race_name')
    prediction = data.get('prediction')

    if not race_name or not isinstance(race_name, str):
        return jsonify({'success': False, 'message': 'race_name required'}), 400
    if not prediction or not isinstance(prediction, dict):
        return jsonify({'success': False, 'message': 'prediction payload required'}), 400

    race_lock = _get_race_lock(race_name)
    with race_lock:
        _save_cache(race_name, prediction)

    return jsonify({'success': True, 'message': f'Prediction saved for {race_name}'})


@app.route('/predict', methods=['POST'])
def predict_single_driver():
    """Individual driver finishing-position prediction."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}

    # --- Input validation (fixes flaw #4) ---
    driver = data.get('driver', '')
    team = data.get('team', '')
    grid_raw = data.get('grid')

    err = _validate_driver(driver)
    if err:
        return jsonify({'error': err}), 400

    err = _validate_team(team)
    if err:
        return jsonify({'error': err}), 400

    grid, err = _validate_grid(grid_raw)
    if err:
        return jsonify({'error': err}), 400

    try:
        driver_enc = _state.master_predictor.label_encoders['Driver'].transform([driver])[0]
        team_enc = _state.master_predictor.label_encoders['Team'].transform([team])[0]
        X = np.array([[driver_enc, team_enc, grid]])
        # Use the model's own predict() which returns a plain Python int (flaw #10)
        predicted_pos = _state.master_predictor.model.predict(X)
        return jsonify({'predicted_position': predicted_pos})
    except Exception as e:
        app.logger.error(f'predict error: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/simulate_wdc', methods=['POST'])
def simulate_wdc():
    """
    Start a WDC simulation job and return a job_id immediately.
    The simulation runs in a background thread (fixes flaw #5).
    Poll /sim_status/<job_id> for results.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    data = request.get_json(silent=True) or {}

    # --- Input validation (fixes flaw #4) ---
    driver = str(data.get('wdcDriver', ''))
    err = _validate_driver(driver)
    if err:
        return jsonify({'error': err}), 400

    try:
        num_sims = int(data.get('simulations', 1_000_000))
        if not (1_000 <= num_sims <= 10_000_000):
            return jsonify({'error': 'simulations must be between 1,000 and 10,000,000'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'simulations must be an integer'}), 400

    seed_raw = data.get('seed')
    seed = int(seed_raw) if seed_raw is not None else None

    try:
        remaining = _state.simulator.get_remaining_races_info(2025) or [
            'Qatar Grand Prix', 'Abu Dhabi Grand Prix'
        ]
    except Exception:
        remaining = ['Qatar Grand Prix', 'Abu Dhabi Grand Prix']

    job_id = str(uuid.uuid4())
    with _sim_jobs_lock:
        _sim_jobs[job_id] = {'status': 'running'}

    t = threading.Thread(
        target=_run_sim,
        args=(job_id, driver, num_sims, remaining, seed),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id, 'status': 'running'})


@app.route('/sim_status/<job_id>', methods=['GET'])
def sim_status(job_id):
    """Poll for simulation results."""
    with _sim_jobs_lock:
        job = _sim_jobs.get(job_id)

    if job is None:
        return jsonify({'error': 'Unknown job_id'}), 404

    return jsonify(_clean(job))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    initialize_system()
    # use_reloader=False prevents double-init of the global state in debug mode
    app.run(debug=True, port=5000, use_reloader=False)
