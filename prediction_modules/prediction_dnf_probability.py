import numpy as np

class DNFPredictor:
    """Predicts DNF probabilities for drivers"""
    
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
        
    def predict_dnf_risk(self, race_name="Next Race"):
        """
        Predict DNF risk for each driver
        
        Returns:
            dict: DNF risk predictions
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        dnf_predictions = []
        
        for driver in drivers:
            team = self._get_driver_team(driver)
            
            # Calculate DNF factors
            reliability_risk = self._get_team_reliability(team)
            driver_risk = self._get_driver_risk_factor(driver)
            track_risk = np.random.uniform(0.02, 0.08)  # Track-specific risk
            
            total_dnf_probability = (reliability_risk + driver_risk + track_risk) * 100
            total_dnf_probability = min(total_dnf_probability, 35)  # Cap at 35%
            
            dnf_predictions.append({
                'driver': driver,
                'team': team,
                'dnf_probability': round(total_dnf_probability, 2),
                'reliability_risk': round(reliability_risk * 100, 2),
                'driver_risk': round(driver_risk * 100, 2),
                'finish_probability': round(100 - total_dnf_probability, 2)
            })
        
        dnf_predictions.sort(key=lambda x: x['dnf_probability'], reverse=True)
        
        return {
            'race_name': race_name,
            'predictions': dnf_predictions,
            'highest_risk': dnf_predictions[:5],
            'safest_bets': dnf_predictions[-5:]
        }
    
    def _get_team_reliability(self, team):
        """Get team reliability factor (higher = more DNFs)"""
        reliability = {
            'McLaren': 0.03,
            'Ferrari': 0.04,
            'Red Bull Racing': 0.04,
            'Mercedes': 0.05,
            'Aston Martin': 0.07,
            'Alpine': 0.08,
            'Haas F1 Team': 0.08,
            'RB': 0.09,
            'Williams': 0.09,
            'Kick Sauber': 0.10
        }
        return reliability.get(team, 0.08)
    
    def _get_driver_risk_factor(self, driver):
        """Get driver-specific risk (crashes, mistakes)"""
        # More experienced drivers have lower risk (2025 grid)
        experienced_drivers = ['VER', 'HAM', 'ALO', 'LEC', 'RUS', 'NOR', 'SAI']
        if driver in experienced_drivers:
            return np.random.uniform(0.01, 0.03)
        else:
            return np.random.uniform(0.02, 0.05)
    
    def _get_driver_team(self, driver):
        """Get team for driver"""
        mapping = {
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
