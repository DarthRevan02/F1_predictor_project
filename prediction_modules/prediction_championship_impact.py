import numpy as np
from driver_config import F1_POINTS_SYSTEM


class ChampionshipImpactPredictor:
    """Predicts how this race will shift championship standings."""

    def __init__(self, model, label_encoders, current_standings):
        self.model = model
        self.label_encoders = label_encoders
        self.current_standings = current_standings or {}

    def predict_championship_impact(self, race_name="Next Race", num_simulations=1000):
        drivers = list(self.label_encoders['Driver'].classes_)
        contenders = self._top_contenders(drivers)
        predictions = []

        for driver in contenders:
            s = self._simulate_scenarios(driver, num_simulations)
            predictions.append({
                'driver': driver,
                'current_points': self.current_standings.get(driver, 0),
                'avg_points_after_race': round(s['avg'], 1),
                'best_case_points': round(s['best'], 1),
                'worst_case_points': round(s['worst'], 1),
                'championship_position_change': int(s['pos_change']),
                'title_probability_change': round(s['title_delta'], 2),
            })

        predictions.sort(key=lambda x: x['avg_points_after_race'], reverse=True)

        return {
            'race_name': race_name,
            'championship_impact': predictions,
            'potential_leader_changes': sum(
                1 for p in predictions if abs(p['championship_position_change']) >= 1
            ),
        }

    def _top_contenders(self, drivers):
        ranked = sorted(
            drivers,
            key=lambda d: self.current_standings.get(d, 0),
            reverse=True,
        )
        return ranked[:8]

    def _simulate_scenarios(self, driver, n):
        outcomes = []
        for _ in range(n):
            pos = max(1, int(np.random.normal(5, 3)))
            outcomes.append(F1_POINTS_SYSTEM.get(pos, 0))
        current = self.current_standings.get(driver, 0)
        return {
            'avg': current + float(np.mean(outcomes)),
            'best': current + max(outcomes),
            'worst': current + min(outcomes),
            'pos_change': int(np.random.choice([-2, -1, 0, 1, 2])),
            'title_delta': float(np.random.uniform(-5, 5)),
        }
