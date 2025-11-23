import numpy as np

class FastestLapPredictor:
    """Predicts who will set the fastest lap"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_fastest_lap(self, race_name="Next Race"):
        """
        Predict fastest lap setter
        
        Returns:
            dict: Fastest lap predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        fastest_lap_predictions = []
        
        for driver in drivers:
            pace_score = self._calculate_pace_score(driver)
            strategy_factor = self._calculate_strategy_factor(driver)
            
            # Fastest lap probability considers both pace and strategy
            fastest_lap_prob = (pace_score * 0.7 + strategy_factor * 0.3) * 100
            
            fastest_lap_predictions.append({
                'driver': driver,
                'team': self._get_driver_team(driver),
                'fastest_lap_probability': fastest_lap_prob,
                'pace_score': pace_score,
                'likely_to_pit_for_fl': strategy_factor > 0.6
            })
        
        fastest_lap_predictions.sort(key=lambda x: x['fastest_lap_probability'], reverse=True)
        
        return {
            'race_name': race_name,
            'predictions': fastest_lap_predictions[:10],
            'most_likely': fastest_lap_predictions[0]
        }
    
    def _calculate_pace_score(self, driver):
        """Calculate raw pace ability"""
        pace_ratings = {
            # 2025 Season pace rankings
            'NOR': 0.98, 'VER': 0.97, 'LEC': 0.96, 'HAM': 0.95,
            'PIA': 0.94, 'RUS': 0.92, 'SAI': 0.90, 'ALO': 0.89,
            'GAS': 0.87, 'LAW': 0.85, 'STR': 0.84, 'OCO': 0.83
        }
        return pace_ratings.get(driver, 0.75) + np.random.uniform(-0.05, 0.05)
    
    def _calculate_strategy_factor(self, driver):
        """Calculate likelihood of pitting for fastest lap"""
        # Top teams more likely to go for fastest lap point
        top_teams = ['NOR', 'VER', 'LEC', 'HAM', 'PIA', 'RUS', 'SAI']
        if driver in top_teams:
            return np.random.uniform(0.6, 0.9)
        else:
            return np.random.uniform(0.2, 0.5)
    
    def _get_driver_team(self, driver):
        """Get team for driver"""
        mapping = {
            'HAM': 'Ferrari', 'LEC': 'Ferrari',
            'VER': 'Red Bull Racing', 'LAW': 'Red Bull Racing',
            'NOR': 'McLaren', 'PIA': 'McLaren',
            'RUS': 'Mercedes', 'ANT': 'Mercedes'
        }
        return mapping.get(driver, 'Unknown')