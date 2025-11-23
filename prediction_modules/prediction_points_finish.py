import numpy as np

class PointsFinishPredictor:
    """Predicts which drivers will score points"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_points_finishers(self, race_name="Next Race", num_simulations=1000):
        """
        Predict top 10 points-scoring finishers
        
        Returns:
            dict: Points finish predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        points_counts = {driver: 0 for driver in drivers}
        position_distributions = {driver: [] for driver in drivers}
        
        # Run simulations
        for _ in range(num_simulations):
            race_results = self._simulate_race_with_reliability(drivers)
            
            # Count points finishes (top 10)
            for i, driver in enumerate(race_results[:10]):
                points_counts[driver] += 1
                position_distributions[driver].append(i + 1)
        
        # Calculate statistics
        predictions = []
        for driver in drivers:
            if points_counts[driver] > 0:
                predictions.append({
                    'driver': driver,
                    'team': self._get_driver_team(driver),
                    'points_probability': (points_counts[driver] / num_simulations) * 100,
                    'avg_position': np.mean(position_distributions[driver]) if position_distributions[driver] else 0,
                    'best_position': min(position_distributions[driver]) if position_distributions[driver] else 20,
                    'worst_position': max(position_distributions[driver]) if position_distributions[driver] else 20
                })
        
        predictions.sort(key=lambda x: x['points_probability'], reverse=True)
        
        return {
            'race_name': race_name,
            'predictions': predictions,
            'guaranteed_points': [p for p in predictions if p['points_probability'] > 90],
            'likely_points': [p for p in predictions if 50 < p['points_probability'] <= 90]
        }
    
    def _simulate_race_with_reliability(self, drivers):
        """Simulate race with DNF possibilities"""
        driver_performance = {
            # 2025 Performance rankings
            'NOR': 0.95, 'PIA': 0.93, 'LEC': 0.92, 'HAM': 0.91,
            'VER': 0.90, 'RUS': 0.88, 'SAI': 0.85, 'ALO': 0.84
        }
        
        race_results = []
        for driver in drivers:
            # Factor in reliability (5% DNF chance)
            if np.random.random() < 0.05:
                continue  # DNF
            
            base_perf = driver_performance.get(driver, 0.65)
            variance = np.random.uniform(0.85, 1.15)
            race_results.append((driver, base_perf * variance))
        
        race_results.sort(key=lambda x: x[1], reverse=True)
        return [driver for driver, _ in race_results]
    
    def _get_driver_team(self, driver):
        """Get team for driver"""
        return 'Team'  # Simplified
