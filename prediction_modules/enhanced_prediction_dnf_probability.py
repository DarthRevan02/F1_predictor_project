"""
Enhanced DNF Predictor using Historical Reliability Data

Uses actual historical DNF rates instead of estimates
"""

import numpy as np
import pandas as pd

class EnhancedDNFPredictor:
    """
    Predicts DNF probabilities using historical reliability data
    """
    
    def __init__(self, model, label_encoders, historical_data=None):
        self.model = model
        self.label_encoders = label_encoders
        self.historical_data = historical_data
        
        # Calculate actual reliability from historical data
        if historical_data is not None:
            self.team_reliability = self._calculate_team_reliability()
            self.driver_reliability = self._calculate_driver_reliability()
        else:
            # Fallback to estimates
            self.team_reliability = self._get_default_team_reliability()
            self.driver_reliability = {}
    
    def _calculate_team_reliability(self):
        """
        Calculate actual team reliability from historical DNF data
        """
        reliability = {}
        
        for team in self.historical_data['Team'].unique():
            team_data = self.historical_data[self.historical_data['Team'] == team]
            
            if len(team_data) > 0:
                finish_rate = 1.0 - team_data['DidNotFinish'].mean()
                reliability[team] = max(0.80, min(0.98, finish_rate))
            else:
                reliability[team] = 0.90
        
        return reliability
    
    def _calculate_driver_reliability(self):
        """
        Calculate driver-specific reliability (mistakes, crashes)
        """
        reliability = {}
        
        for driver in self.historical_data['Driver'].unique():
            driver_data = self.historical_data[self.historical_data['Driver'] == driver]
            
            if len(driver_data) > 0:
                finish_rate = 1.0 - driver_data['DidNotFinish'].mean()
                reliability[driver] = max(0.85, min(0.99, finish_rate))
            else:
                reliability[driver] = 0.95
        
        return reliability
    
    def _get_default_team_reliability(self):
        """Default reliability if no historical data"""
        return {
            'McLaren': 0.96, 'Ferrari': 0.94, 'Red Bull Racing': 0.95,
            'Mercedes': 0.94, 'Aston Martin': 0.90, 'Alpine': 0.88,
            'Haas F1 Team': 0.87, 'RB': 0.89, 'Williams': 0.86,
            'Kick Sauber': 0.84
        }
    
    def predict_dnf_risk(self, race_name="Next Race"):
        """
        Predict DNF risk using historical reliability data
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        dnf_predictions = []
        
        # Determine circuit type (affects DNF rate)
        circuit_type = self._get_circuit_type(race_name)
        circuit_multiplier = 1.5 if circuit_type == 'street' else 1.0
        
        for driver in drivers:
            team = self._get_driver_team(driver)
            
            # Get historical reliability
            team_finish_rate = self.team_reliability.get(team, 0.90)
            driver_finish_rate = self.driver_reliability.get(driver, 0.95)
            
            # Calculate DNF probability
            team_dnf_rate = (1 - team_finish_rate)
            driver_dnf_rate = (1 - driver_finish_rate)
            
            # Combined DNF probability (not just additive, but probabilistic)
            combined_finish_prob = team_finish_rate * driver_finish_rate
            combined_dnf_prob = 1 - combined_finish_prob
            
            # Apply circuit multiplier
            total_dnf_probability = combined_dnf_prob * circuit_multiplier * 100
            total_dnf_probability = min(total_dnf_probability, 40.0)  # Cap at 40%
            
            dnf_predictions.append({
                'driver': driver,
                'team': team,
                'dnf_probability': round(total_dnf_probability, 2),
                'reliability_risk': round(team_dnf_rate * 100, 2),
                'driver_risk': round(driver_dnf_rate * 100, 2),
                'finish_probability': round(100 - total_dnf_probability, 2),
                'circuit_type': circuit_type,
                'historical_finish_rate': round(combined_finish_prob * 100, 1)
            })
        
        dnf_predictions.sort(key=lambda x: x['dnf_probability'], reverse=True)
        
        return {
            'race_name': race_name,
            'predictions': dnf_predictions,
            'highest_risk': dnf_predictions[:5],
            'safest_bets': dnf_predictions[-5:],
            'data_source': 'Historical data' if self.historical_data is not None else 'Estimates',
            'circuit_type': circuit_type,
            'average_dnf_rate': round(np.mean([p['dnf_probability'] for p in dnf_predictions]), 1)
        }
    
    def _get_circuit_type(self, race_name):
        """Determine if circuit is street or permanent"""
        street_circuits = [
            'Monaco Grand Prix', 'Singapore Grand Prix', 
            'Las Vegas Grand Prix', 'Azerbaijan Grand Prix',
            'Saudi Arabian Grand Prix', 'Miami Grand Prix'
        ]
        
        return 'street' if race_name in street_circuits else 'permanent'
    
    def _get_driver_team(self, driver):
        """Get team for driver (2025 grid)"""
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
    
    def get_reliability_report(self):
        """
        Generate a comprehensive reliability report
        """
        if self.historical_data is None:
            return {"error": "No historical data available"}
        
        print("\n" + "="*60)
        print("🔧 RELIABILITY ANALYSIS REPORT")
        print("="*60)
        
        print("\n📊 Team Reliability (Historical Finish Rates):\n")
        sorted_teams = sorted(
            self.team_reliability.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (team, reliability) in enumerate(sorted_teams, 1):
            dnf_rate = (1 - reliability) * 100
            bar = "█" * int(reliability * 50)
            print(f"  {rank:2d}. {team:25s} {bar} {reliability:.1%} (DNF: {dnf_rate:.1f}%)")
        
        if self.driver_reliability:
            print("\n👤 Driver Reliability (Top 10 Most Reliable):\n")
            sorted_drivers = sorted(
                self.driver_reliability.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for rank, (driver, reliability) in enumerate(sorted_drivers, 1):
                dnf_rate = (1 - reliability) * 100
                print(f"  {rank:2d}. {driver}: {reliability:.1%} (DNF: {dnf_rate:.1f}%)")
        
        print("\n" + "="*60 + "\n")
        
        return {
            'team_reliability': self.team_reliability,
            'driver_reliability': self.driver_reliability,
            'most_reliable_team': sorted_teams[0][0],
            'least_reliable_team': sorted_teams[-1][0]
        }