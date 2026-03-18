import numpy as np
from driver_config import TEAM_RELIABILITY


_TEAM_STRENGTH = {
    'McLaren': 0.93, 'Ferrari': 0.91, 'Red Bull Racing': 0.90,
    'Mercedes': 0.87, 'Aston Martin': 0.77, 'Alpine': 0.70,
    'Haas F1 Team': 0.68, 'RB': 0.67, 'Williams': 0.64, 'Kick Sauber': 0.60,
}


class TeamPerformancePredictor:
    """Predicts team-level performance metrics."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_team_performance(self, race_name="Next Race"):
        teams = list(_TEAM_STRENGTH.keys())
        predictions = []

        for team in teams:
            s = _TEAM_STRENGTH[team]
            if s > 0.85:
                pts = float(np.random.uniform(30, 43))
            elif s > 0.70:
                pts = float(np.random.uniform(15, 30))
            else:
                pts = float(np.random.uniform(0, 15))

            predictions.append({
                'team': team,
                'expected_points': round(pts, 1),
                'best_driver_position': int(max(1, round((1 - s) * 15 + np.random.randint(-2, 3)))),
                'both_cars_in_points_prob': round(min(95.0, s * 100 + np.random.uniform(-10, 10)), 1),
                'podium_probability': round(min(95.0, max(0.0, (s - 0.75) * 200 + np.random.uniform(-10, 10))), 1),
                'performance_rating': round(s * 100, 1),
                'reliability_factor': round((1 - TEAM_RELIABILITY.get(team, 0.08)) * 100, 1),
            })

        predictions.sort(key=lambda x: x['expected_points'], reverse=True)

        return {
            'race_name': race_name,
            'team_predictions': predictions,
            'top_team': predictions[0],
            'podium_contenders': [t for t in predictions if t['podium_probability'] > 50],
        }
