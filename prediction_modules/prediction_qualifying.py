import numpy as np
from driver_config import DRIVER_TEAM_MAP, DRIVER_QUALI_SKILL, TEAM_QUALI_STRENGTH


class QualifyingPredictor:
    """Predicts qualifying session grid positions."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_qualifying(self, race_name="Next Race"):
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []

        for driver in drivers:
            team = DRIVER_TEAM_MAP.get(driver, 'Unknown')
            score = self._quali_score(driver, team)
            predictions.append({
                'driver': driver,
                'team': team,
                'predicted_grid': 0,
                'performance_score': score,
                'q3_probability': self._q3_probability(score),
            })

        predictions.sort(key=lambda x: x['performance_score'], reverse=True)
        for i, p in enumerate(predictions):
            p['predicted_grid'] = i + 1

        return {
            'race_name': race_name,
            'predicted_grid': predictions,
            'pole_position': predictions[0],
            'q3_contenders': predictions[:10],
        }

    def _quali_score(self, driver, team):
        team_s = TEAM_QUALI_STRENGTH.get(team, 0.65)
        driver_s = DRIVER_QUALI_SKILL.get(driver, 0.75)
        return (team_s * 0.6 + driver_s * 0.4) * np.random.uniform(0.95, 1.05)

    def _q3_probability(self, score):
        if score > 0.85:
            return 95.0
        if score > 0.75:
            return 70.0
        if score > 0.68:
            return 40.0
        return 15.0
