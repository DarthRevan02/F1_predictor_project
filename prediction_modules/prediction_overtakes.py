import numpy as np
from driver_config import DRIVER_TEAM_MAP, DRIVER_PERFORMANCE

# Overtaking skill roughly mirrors raw performance but with its own weighting
_OVERTAKING = {
    'VER': 0.95, 'HAM': 0.94, 'ALO': 0.93, 'NOR': 0.92,
    'LEC': 0.90, 'PIA': 0.89, 'SAI': 0.88, 'RUS': 0.87,
    'GAS': 0.85, 'OCO': 0.84, 'STR': 0.83, 'LAW': 0.82,
}

_DEFENSIVE = {
    'VER': 0.96, 'HAM': 0.93, 'ALO': 0.94, 'LEC': 0.91,
    'NOR': 0.89, 'SAI': 0.87, 'RUS': 0.86, 'PIA': 0.88,
}


class OvertakePredictor:
    """Predicts per-driver overtaking statistics."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_position_changes(self, race_name="Next Race"):
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []

        for driver in drivers:
            ot = _OVERTAKING.get(driver, DRIVER_PERFORMANCE.get(driver, 0.75))
            df = _DEFENSIVE.get(driver, DRIVER_PERFORMANCE.get(driver, 0.75))

            gained = max(0.0, ot * 4 * np.random.uniform(0.8, 1.2))
            lost = max(0.0, (1 - df) * 3 * np.random.uniform(0.8, 1.2))

            predictions.append({
                'driver': driver,
                'team': DRIVER_TEAM_MAP.get(driver, 'Unknown'),
                'expected_overtakes_made': round(gained, 1),
                'expected_positions_lost': round(lost, 1),
                'net_position_change': round(gained - lost, 1),
                'overtaking_rating': round(ot * 100, 1),
                'defensive_rating': round(df * 100, 1),
            })

        predictions.sort(key=lambda x: x['expected_overtakes_made'], reverse=True)

        return {
            'race_name': race_name,
            'predictions': predictions,
            'best_overtakers': predictions[:5],
            'most_improved': sorted(
                predictions, key=lambda x: x['net_position_change'], reverse=True
            )[:5],
        }
