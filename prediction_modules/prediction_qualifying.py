import numpy as np

class QualifyingPredictor:
    """Predicts qualifying session results"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        self.team_quali_strength = {
            'McLaren': 0.95,
            'Ferrari': 0.93, 
            'Red Bull Racing': 0.91,
            'Mercedes': 0.89,
            'Aston Martin': 0.80,
            'Alpine': 0.75,
            'Haas F1 Team': 0.72,
            'RB': 0.70,
            'Williams': 0.68,
            'Kick Sauber': 0.63
        }
        
    def predict_qualifying(self, race_name="Next Race"):
        """
        Predict qualifying grid positions
        
        Returns:
            dict: Predicted starting grid
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        quali_predictions = []
        
        for driver in drivers:
            team = self._get_driver_team(driver)
            quali_performance = self._calculate_quali_performance(driver, team)
            
            quali_predictions.append({
                'driver': driver,
                'team': team,
                'predicted_grid': 0,  # Will be set after sorting
                'performance_score': quali_performance,
                'q3_probability': self._calculate_q3_probability(quali_performance)
            })
        
        # Sort by performance and assign grid positions
        quali_predictions.sort(key=lambda x: x['performance_score'], reverse=True)
        for i, pred in enumerate(quali_predictions):
            pred['predicted_grid'] = i + 1
        
        return {
            'race_name': race_name,
            'predicted_grid': quali_predictions,
            'pole_position': quali_predictions[0],
            'q3_contenders': quali_predictions[:10]
        }
    
    def _calculate_quali_performance(self, driver, team):
        """Calculate qualifying performance score"""
        driver_skill = {
            # 2025 Season qualifying specialists
            'NOR': 0.98, 'LEC': 0.97, 'PIA': 0.95, 'HAM': 0.94,
            'VER': 0.93, 'RUS': 0.92, 'SAI': 0.89, 'ALO': 0.88,
            'GAS': 0.87, 'STR': 0.85, 'LAW': 0.84, 'OCO': 0.83,
            'TSU': 0.82, 'ALB': 0.81, 'HUL': 0.80, 'ANT': 0.78
        }
        
        team_strength = self.team_quali_strength.get(team, 0.65)
        driver_strength = driver_skill.get(driver, 0.75)
        
        # Add random variance for track-specific performance
        variance = np.random.uniform(0.95, 1.05)
        
        return (team_strength * 0.6 + driver_strength * 0.4) * variance
    
    def _calculate_q3_probability(self, performance_score):
        """Calculate probability of reaching Q3"""
        if performance_score > 0.85:
            return 95.0
        elif performance_score > 0.75:
            return 70.0
        elif performance_score > 0.68:
            return 40.0
        else:
            return 15.0
    
    def _get_driver_team(self, driver):
        """Get team for driver"""
        mapping = {
            # 2025 Grid
            'HAM': 'Ferrari', 'LEC': 'Ferrari',
            'VER': 'Red Bull Racing', 'LAW': 'Red Bull Racing',
            'NOR': 'McLaren', 'PIA': 'McLaren',
            'RUS': 'Mercedes', 'ANT': 'Mercedes',
            'ALO': 'Aston Martin', 'STR': 'Aston Martin',
            'GAS': 'Alpine', 'DOR': 'Alpine',
            'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
            'TSU': 'RB', 'HAD': 'RB',
            'SAI': 'Williams', 'ALB': 'Williams',
            'HUL': 'Kick Sauber', 'BOR': 'Kick Sauber'
        }
        return mapping.get(driver, 'Unknown')
