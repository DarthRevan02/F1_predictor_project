import numpy as np

class TeamPerformancePredictor:
    """Predicts team-level performance metrics"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_team_performance(self, race_name="Next Race"):
        """
        Predict how each team will perform
        
        Returns:
            dict: Team performance predictions
        """
        teams = ['McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes', 
                'Aston Martin', 'Alpine', 'Haas F1 Team', 'RB', 'Williams', 'Kick Sauber']
        
        team_predictions = []
        
        for team in teams:
            performance = self._calculate_team_performance(team)
            
            team_predictions.append({
                'team': team,
                'expected_points': performance['expected_points'],
                'best_driver_position': performance['best_position'],
                'both_cars_in_points_prob': performance['both_points_prob'],
                'podium_probability': performance['podium_prob'],
                'performance_rating': performance['rating'],
                'reliability_factor': performance['reliability']
            })
        
        team_predictions.sort(key=lambda x: x['expected_points'], reverse=True)
        
        return {
            'race_name': race_name,
            'team_predictions': team_predictions,
            'top_team': team_predictions[0],
            'podium_contenders': [t for t in team_predictions if t['podium_probability'] > 50]
        }
    
    def _calculate_team_performance(self, team):
        """Calculate comprehensive team performance metrics"""
        team_strength = {
            'McLaren': 0.93,
            'Ferrari': 0.91,
            'Red Bull Racing': 0.90,
            'Mercedes': 0.87,
            'Aston Martin': 0.77,
            'Alpine': 0.70,
            'Haas F1 Team': 0.68,
            'RB': 0.67,
            'Williams': 0.64,
            'Kick Sauber': 0.60
        }
        
        strength = team_strength.get(team, 0.60)
        
        # Calculate expected points (both cars combined)
        expected_points = self._calculate_expected_team_points(strength)
        
        # Calculate best driver position
        best_position = max(1, int((1 - strength) * 15 + np.random.randint(-2, 3)))
        
        # Probability both cars finish in points
        both_points_prob = min(95, strength * 100 + np.random.uniform(-10, 10))
        
        # Podium probability
        podium_prob = min(95, max(0, (strength - 0.75) * 200 + np.random.uniform(-10, 10)))
        
        # Reliability factor
        reliability = self._get_team_reliability(team)
        
        return {
            'expected_points': round(expected_points, 1),
            'best_position': best_position,
            'both_points_prob': round(both_points_prob, 1),
            'podium_prob': round(podium_prob, 1),
            'rating': round(strength * 100, 1),
            'reliability': round(reliability * 100, 1)
        }
    
    def _calculate_expected_team_points(self, strength):
        """Calculate expected points for both cars"""
        # Top teams can score 40+ points, mid-field 10-20, back-markers 0-5
        if strength > 0.85:
            return np.random.uniform(30, 43)
        elif strength > 0.70:
            return np.random.uniform(15, 30)
        else:
            return np.random.uniform(0, 15)
    
    def _get_team_reliability(self, team):
        """Get team reliability score"""
        reliability_scores = {
            'McLaren': 0.95,
            'Ferrari': 0.93,
            'Red Bull Racing': 0.94,
            'Mercedes': 0.93,
            'Aston Martin': 0.89,
            'Alpine': 0.87,
            'Haas F1 Team': 0.86,
            'RB': 0.88,
            'Williams': 0.85,
            'Kick Sauber': 0.83
        }
        return reliability_scores.get(team, 0.85)