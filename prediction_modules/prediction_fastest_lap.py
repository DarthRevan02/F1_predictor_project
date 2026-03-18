import numpy as np
from driver_config import DRIVER_TEAM_MAP, DRIVER_PERFORMANCE


# Drivers whose teams typically target the fastest-lap bonus point
_FL_TEAMS = {'McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes'}


class FastestLapPredictor:
    """Predicts fastest lap setter."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_fastest_lap(self, race_name="Next Race"):
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []

        for driver in drivers:
            team = DRIVER_TEAM_MAP.get(driver, 'Unknown')
            pace = DRIVER_PERFORMANCE.get(driver, 0.65) + np.random.uniform(-0.05, 0.05)
            strategy = np.random.uniform(0.6, 0.9) if team in _FL_TEAMS else np.random.uniform(0.2, 0.5)
            prob = (pace * 0.7 + strategy * 0.3) * 100

            predictions.append({
                'driver': driver,
                'team': team,
                'fastest_lap_probability': prob,
                'pace_score': round(pace, 3),
                'likely_to_pit_for_fl': strategy > 0.6,
            })

        predictions.sort(key=lambda x: x['fastest_lap_probability'], reverse=True)

        return {
            'race_name': race_name,
            'predictions': predictions[:10],
            'most_likely': predictions[0],
        }
