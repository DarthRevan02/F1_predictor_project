import numpy as np
from driver_config import DRIVER_TEAM_MAP, DRIVER_PERFORMANCE


class PointsFinishPredictor:
    """Predicts which drivers will score championship points."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_points_finishers(self, race_name="Next Race", num_simulations=1000):
        drivers = list(self.label_encoders['Driver'].classes_)
        points_counts = {d: 0 for d in drivers}
        position_dist = {d: [] for d in drivers}

        for _ in range(num_simulations):
            results = self._simulate_race(drivers)
            for i, driver in enumerate(results[:10]):
                points_counts[driver] += 1
                position_dist[driver].append(i + 1)

        predictions = []
        for driver in drivers:
            if points_counts[driver] > 0:
                pos_list = position_dist[driver]
                predictions.append({
                    'driver': driver,
                    'team': DRIVER_TEAM_MAP.get(driver, 'Unknown'),
                    'points_probability': (points_counts[driver] / num_simulations) * 100,
                    'avg_position': round(float(np.mean(pos_list)), 2),
                    'best_position': int(min(pos_list)),
                    'worst_position': int(max(pos_list)),
                })

        predictions.sort(key=lambda x: x['points_probability'], reverse=True)

        return {
            'race_name': race_name,
            'predictions': predictions,
            'guaranteed_points': [p for p in predictions if p['points_probability'] > 90],
            'likely_points': [p for p in predictions if 50 < p['points_probability'] <= 90],
        }

    def _simulate_race(self, drivers):
        results = []
        for driver in drivers:
            if np.random.random() < 0.05:  # 5% DNF
                continue
            base = DRIVER_PERFORMANCE.get(driver, 0.65)
            results.append((driver, base * np.random.uniform(0.85, 1.15)))
        results.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in results]
