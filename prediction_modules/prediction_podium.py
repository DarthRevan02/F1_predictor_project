import numpy as np
from driver_config import DRIVER_PERFORMANCE


class PodiumPredictor:
    """Monte Carlo podium probability predictor."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_podium(self, race_name="Next Race", num_simulations=1000):
        drivers = list(self.label_encoders['Driver'].classes_)
        podium_counts = {d: {'P1': 0, 'P2': 0, 'P3': 0, 'total': 0} for d in drivers}

        for _ in range(num_simulations):
            results = self._simulate_single_race(drivers)
            for pos, driver in enumerate(results[:3]):
                key = f'P{pos+1}'
                podium_counts[driver][key] += 1
                podium_counts[driver]['total'] += 1

        predictions = []
        for driver, counts in podium_counts.items():
            if counts['total'] > 0:
                predictions.append({
                    'driver': driver,
                    'podium_probability': (counts['total'] / num_simulations) * 100,
                    'win_probability': (counts['P1'] / num_simulations) * 100,
                    'p2_probability': (counts['P2'] / num_simulations) * 100,
                    'p3_probability': (counts['P3'] / num_simulations) * 100,
                })

        predictions.sort(key=lambda x: x['podium_probability'], reverse=True)

        return {
            'race_name': race_name,
            'top_contenders': predictions[:6],
            'most_likely_winner': predictions[0] if predictions else None,
        }

    def _simulate_single_race(self, drivers):
        """Single race simulation using centralised performance ratings."""
        results = []
        for driver in drivers:
            base = DRIVER_PERFORMANCE.get(driver, 0.60)
            results.append((driver, base * np.random.uniform(0.85, 1.15)))
        results.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in results]
