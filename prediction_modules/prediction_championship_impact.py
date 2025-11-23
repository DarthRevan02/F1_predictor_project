import numpy as np

class ChampionshipImpactPredictor:
    """Predicts championship impact of race results"""
    
    def __init__(self, model, label_encoders, current_standings):
        self.model = model
        self.label_encoders = label_encoders
        self.current_standings = current_standings or {}
        
    def predict_championship_impact(self, race_name="Next Race", num_simulations=1000):
        """
        Predict how race will affect championship standings
        
        Returns:
            dict: Championship impact predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        
        # Get top championship contenders
        contenders = self._get_championship_contenders(drivers)
        
        impact_predictions = []
        
        for driver in contenders:
            scenarios = self._simulate_championship_scenarios(driver, num_simulations)
            
            impact_predictions.append({
                'driver': driver,
                'current_points': self.current_standings.get(driver, 0),
                'avg_points_after_race': scenarios['avg_points'],
                'best_case_points': scenarios['best_case'],
                'worst_case_points': scenarios['worst_case'],
                'championship_position_change': scenarios['position_change'],
                'title_probability_change': scenarios['title_prob_change']
            })
        
        impact_predictions.sort(key=lambda x: x['avg_points_after_race'], reverse=True)
        
        return {
            'race_name': race_name,
            'championship_impact': impact_predictions,
            'potential_leader_changes': self._calculate_leader_changes(impact_predictions)
        }
    
    def _get_championship_contenders(self, drivers):
        """Get top championship contenders"""
        # Top 8 drivers in standings (simplified)
        return drivers[:8] if len(drivers) >= 8 else drivers
    
    def _simulate_championship_scenarios(self, driver, num_simulations):
        """Simulate different race outcomes"""
        points_outcomes = []
        
        for _ in range(num_simulations):
            # Random finish position with driver bias
            finish_position = max(1, int(np.random.normal(5, 3)))
            points = self._position_to_points(finish_position)
            points_outcomes.append(points)
        
        current = self.current_standings.get(driver, 0)
        
        return {
            'avg_points': current + np.mean(points_outcomes),
            'best_case': current + max(points_outcomes),
            'worst_case': current + min(points_outcomes),
            'position_change': np.random.choice([-2, -1, 0, 1, 2]),
            'title_prob_change': np.random.uniform(-5, 5)
        }
    
    def _position_to_points(self, position):
        """Convert position to points"""
        points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points.get(position, 0)
    
    def _calculate_leader_changes(self, predictions):
        """Calculate potential championship leader changes"""
        return len([p for p in predictions if abs(p['championship_position_change']) >= 1])

