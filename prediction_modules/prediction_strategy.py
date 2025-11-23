import numpy as np

class StrategyPredictor:
    """Predicts race strategies and pit stop windows"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_race_strategy(self, race_name="Next Race", race_distance=58):
        """
        Predict pit stop strategies for drivers
        
        Args:
            race_name: Name of the race
            race_distance: Number of laps
            
        Returns:
            dict: Strategy predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)[:10]  # Top 10
        strategy_predictions = []
        
        for driver in drivers:
            team = self._get_driver_team(driver)
            strategy = self._predict_driver_strategy(driver, team, race_distance)
            
            strategy_predictions.append({
                'driver': driver,
                'team': team,
                'predicted_strategy': strategy['strategy_type'],
                'num_pit_stops': strategy['num_stops'],
                'first_stop_lap': strategy['first_stop'],
                'tire_compounds': strategy['compounds'],
                'strategy_risk': strategy['risk_level'],
                'undercut_probability': strategy['undercut_prob']
            })
        
        return {
            'race_name': race_name,
            'race_distance': race_distance,
            'strategy_predictions': strategy_predictions,
            'most_aggressive': max(strategy_predictions, key=lambda x: x['strategy_risk']),
            'most_conservative': min(strategy_predictions, key=lambda x: x['strategy_risk'])
        }
    
    def _predict_driver_strategy(self, driver, team, race_distance):
        """Predict specific strategy for a driver"""
        # Determine number of stops (most races are 1-2 stop)
        if race_distance < 50:
            num_stops = np.random.choice([1, 2], p=[0.7, 0.3])
        else:
            num_stops = np.random.choice([1, 2, 3], p=[0.4, 0.5, 0.1])
        
        # Determine strategy type
        if num_stops == 1:
            strategy_type = "One-Stop"
            compounds = ["Medium", "Hard"]
            first_stop = np.random.randint(race_distance // 3, race_distance // 2)
            risk_level = 30
        elif num_stops == 2:
            strategy_type = "Two-Stop"
            compounds = ["Soft", "Medium", "Hard"]
            first_stop = np.random.randint(race_distance // 4, race_distance // 3)
            risk_level = 50
        else:
            strategy_type = "Three-Stop (Aggressive)"
            compounds = ["Soft", "Soft", "Medium", "Hard"]
            first_stop = np.random.randint(10, 20)
            risk_level = 75
        
        # Top teams more likely to attempt undercuts
        undercut_prob = 70 if team in ['McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes'] else 40
        
        return {
            'strategy_type': strategy_type,
            'num_stops': num_stops,
            'first_stop': first_stop,
            'compounds': compounds,
            'risk_level': risk_level,
            'undercut_prob': undercut_prob
        }
    
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
