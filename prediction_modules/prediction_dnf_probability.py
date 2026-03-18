import numpy as np
from driver_config import DRIVER_TEAM_MAP, TEAM_RELIABILITY

_EXPERIENCED = {'VER', 'HAM', 'ALO', 'LEC', 'RUS', 'NOR', 'SAI'}


class DNFPredictor:
    """Predicts DNF risk for each driver."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_dnf_risk(self, race_name="Next Race"):
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []

        for driver in drivers:
            team = DRIVER_TEAM_MAP.get(driver, 'Unknown')
            rel_risk = TEAM_RELIABILITY.get(team, 0.08)
            drv_risk = (
                np.random.uniform(0.01, 0.03) if driver in _EXPERIENCED
                else np.random.uniform(0.02, 0.05)
            )
            track_risk = np.random.uniform(0.02, 0.08)
            total = min((rel_risk + drv_risk + track_risk) * 100, 35.0)

            predictions.append({
                'driver': driver,
                'team': team,
                'dnf_probability': round(total, 2),
                'reliability_risk': round(rel_risk * 100, 2),
                'driver_risk': round(drv_risk * 100, 2),
                'finish_probability': round(100 - total, 2),
            })

        predictions.sort(key=lambda x: x['dnf_probability'], reverse=True)

        return {
            'race_name': race_name,
            'predictions': predictions,
            'highest_risk': predictions[:5],
            'safest_bets': predictions[-5:],
        }
