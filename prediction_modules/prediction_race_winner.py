import numpy as np
import pandas as pd

class RaceWinnerPredictor:
    """Predicts the winner of a specific race"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        self.driver_team_mapping = self._initialize_driver_teams()
        
    def _initialize_driver_teams(self):
        """2025 F1 Driver-Team mappings"""
        return {
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
    
    def predict_race_winner(self, race_name="Next Race"):
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []
        
        # Get raw predictions for all drivers
        for driver in drivers:
            team = self.driver_team_mapping.get(driver, 'Unknown')
            if team == 'Unknown':
                continue
                
            estimated_grid = self._estimate_grid_position(driver, team)
            predicted_pos = self._predict_position(driver, team, estimated_grid)
            
            predictions.append({
                'driver': driver,
                'team': team,
                'estimated_grid': estimated_grid,
                'raw_predicted_position': predicted_pos,  # Keep raw value for sorting
            })
        
        # Sort by raw predicted position to get proper ranking
        predictions.sort(key=lambda x: x['raw_predicted_position'])
        
        # Assign unique positions (P1, P2, P3, etc.) and calculate points
        for rank, pred in enumerate(predictions, start=1):
            pred['predicted_position'] = rank  # Assign unique position
            pred['predicted_points'] = self._position_to_points(rank)
            pred['win_probability'] = self._calculate_win_probability(pred['raw_predicted_position'])
            # Remove raw position from output
            del pred['raw_predicted_position']
        
        return {
            'race_name': race_name,
            'predictions': predictions[:10],  # Top 10
            'winner': predictions[0] if predictions else {},
            'podium': predictions[:3] if len(predictions) >= 3 else predictions
        }
    
    def _predict_position(self, driver, team, grid_position):
        """Predict finishing position with error handling"""
        try:
            driver_encoded = self.label_encoders['Driver'].transform([driver])[0]
            team_encoded = self.label_encoders['Team'].transform([team])[0]
            X_pred = np.array([[driver_encoded, team_encoded, grid_position]])
            
            prediction = self.model.predict(X_pred)
            
            # Handle both array and scalar returns
            if isinstance(prediction, (list, np.ndarray)):
                return float(prediction[0])
            return float(prediction)
        except Exception as e:
            print(f"Error predicting for {driver}: {e}")
            return 10.0  # Default mid-field position
    
    def _estimate_grid_position(self, driver, team):
        team_performance = {
            'McLaren': 2, 'Ferrari': 3, 'Red Bull Racing': 4,
            'Mercedes': 6, 'Aston Martin': 9, 'Alpine': 12,
            'Haas F1 Team': 13, 'RB': 15, 'Williams': 17,
            'Kick Sauber': 19
        }
        base_position = team_performance.get(team, 10)
        return max(1, base_position + np.random.randint(-2, 3))
    
    def _position_to_points(self, position):
        points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points.get(position, 0)
    
    def _calculate_win_probability(self, predicted_position):
        if predicted_position <= 1.5:
            return min(95, 100 - (predicted_position - 1) * 30)
        else:
            return max(0, 40 - (predicted_position - 2) * 8)