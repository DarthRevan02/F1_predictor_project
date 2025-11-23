import numpy as np

class OvertakePredictor:
    """Predicts overtaking statistics for races"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_position_changes(self, race_name="Next Race"):
        """
        Predict position changes and overtaking action
        
        Returns:
            dict: Overtaking predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        overtake_predictions = []
        
        for driver in drivers:
            overtaking_skill = self._get_overtaking_skill(driver)
            defensive_skill = self._get_defensive_skill(driver)
            
            expected_positions_gained = self._calculate_positions_gained(driver, overtaking_skill)
            expected_positions_lost = self._calculate_positions_lost(driver, defensive_skill)
            
            net_change = expected_positions_gained - expected_positions_lost
            
            overtake_predictions.append({
                'driver': driver,
                'team': self._get_driver_team(driver),
                'expected_overtakes_made': round(expected_positions_gained, 1),
                'expected_positions_lost': round(expected_positions_lost, 1),
                'net_position_change': round(net_change, 1),
                'overtaking_rating': round(overtaking_skill * 100, 1),
                'defensive_rating': round(defensive_skill * 100, 1)
            })
        
        overtake_predictions.sort(key=lambda x: x['expected_overtakes_made'], reverse=True)
        
        return {
            'race_name': race_name,
            'predictions': overtake_predictions,
            'best_overtakers': overtake_predictions[:5],
            'most_improved': sorted(overtake_predictions, key=lambda x: x['net_position_change'], reverse=True)[:5]
        }
    
    def _get_overtaking_skill(self, driver):
        """Rate driver's overtaking ability"""
        overtaking_ratings = {
            # 2025 best overtakers
            'VER': 0.95, 'HAM': 0.94, 'NOR': 0.92, 'ALO': 0.93,
            'LEC': 0.90, 'SAI': 0.88, 'RUS': 0.87, 'PIA': 0.89,
            'GAS': 0.85, 'OCO': 0.84, 'STR': 0.83, 'LAW': 0.82
        }
        return overtaking_ratings.get(driver, 0.75)
    
    def _get_defensive_skill(self, driver):
        """Rate driver's defensive ability"""
        defensive_ratings = {
            'VER': 0.96, 'HAM': 0.93, 'ALO': 0.94, 'LEC': 0.91,
            'NOR': 0.89, 'SAI': 0.87, 'RUS': 0.86, 'PIA': 0.88
        }
        return defensive_ratings.get(driver, 0.75)
    
    def _calculate_positions_gained(self, driver, skill):
        """Calculate expected positions gained"""
        # Top drivers gain more positions through strategy and pace
        base_gain = skill * 4  # Max 4 positions for best drivers
        variance = np.random.uniform(0.8, 1.2)
        return max(0, base_gain * variance)
    
    def _calculate_positions_lost(self, driver, defensive_skill):
        """Calculate expected positions lost"""
        # Better defenders lose fewer positions
        base_loss = (1 - defensive_skill) * 3
        variance = np.random.uniform(0.8, 1.2)
        return max(0, base_loss * variance)
    
    def _get_driver_team(self, driver):
        """Get team for driver"""
        return 'Team'