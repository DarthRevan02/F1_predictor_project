import numpy as np

class PodiumPredictor:
    """Predicts podium finishers for races"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_podium(self, race_name="Next Race", num_simulations=1000):
        """
        Predict podium finishers using Monte Carlo approach
        
        Returns:
            dict: Podium predictions with probabilities
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        podium_counts = {driver: {'P1': 0, 'P2': 0, 'P3': 0, 'total': 0} for driver in drivers}
        
        # Run multiple simulations
        for _ in range(num_simulations):
            # Simulate race with variance
            race_results = self._simulate_single_race(drivers)
            
            # Count podium finishes
            if len(race_results) >= 3:
                podium_counts[race_results[0]]['P1'] += 1
                podium_counts[race_results[0]]['total'] += 1
                
                podium_counts[race_results[1]]['P2'] += 1
                podium_counts[race_results[1]]['total'] += 1
                
                podium_counts[race_results[2]]['P3'] += 1
                podium_counts[race_results[2]]['total'] += 1
        
        # Calculate probabilities
        podium_predictions = []
        for driver, counts in podium_counts.items():
            if counts['total'] > 0:
                podium_predictions.append({
                    'driver': driver,
                    'podium_probability': (counts['total'] / num_simulations) * 100,
                    'win_probability': (counts['P1'] / num_simulations) * 100,
                    'p2_probability': (counts['P2'] / num_simulations) * 100,
                    'p3_probability': (counts['P3'] / num_simulations) * 100
                })
        
        podium_predictions.sort(key=lambda x: x['podium_probability'], reverse=True)
        
        return {
            'race_name': race_name,
            'top_contenders': podium_predictions[:6],
            'most_likely_winner': podium_predictions[0] if podium_predictions else None
        }
    
    def _simulate_single_race(self, drivers):
        """Simulate a single race outcome"""
        driver_performance = {
            # 2025 Season expected performance
            'NOR': 0.95, 'PIA': 0.93, 'LEC': 0.92, 'HAM': 0.91,
            'VER': 0.90, 'RUS': 0.88, 'SAI': 0.85, 'ALO': 0.84,
            'GAS': 0.82, 'STR': 0.81, 'LAW': 0.79, 'OCO': 0.78,
            'TSU': 0.76, 'ALB': 0.75, 'HUL': 0.73, 'ANT': 0.72,
            'BEA': 0.70, 'DOR': 0.69, 'HAD': 0.68, 'BOR': 0.65
        }
        
        race_performance = []
        for driver in drivers:
            base_perf = driver_performance.get(driver, 0.60)
            variance = np.random.uniform(0.85, 1.15)
            race_performance.append((driver, base_perf * variance))
        
        race_performance.sort(key=lambda x: x[1], reverse=True)
        return [driver for driver, _ in race_performance]