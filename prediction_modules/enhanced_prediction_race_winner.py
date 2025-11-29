"""
Enhanced Race Winner Predictor using Advanced ML Models

This version uses the full feature set and trained ML models instead of simple heuristics
"""

import numpy as np
import pandas as pd

class EnhancedRaceWinnerPredictor:
    """
    Predicts race winners using advanced ML models and engineered features
    """
    
    def __init__(self, model, preprocessor, label_encoders, historical_data=None):
        self.model = model  # Enhanced ML model
        self.preprocessor = preprocessor  # Feature engineering
        self.label_encoders = label_encoders
        self.historical_data = historical_data
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
        """
        Predict race winner using ML model with full feature set
        """
        drivers = list(self.label_encoders['Driver'].classes_)
        predictions = []
        
        # Get circuit characteristics
        circuit_info = self._get_circuit_info(race_name)
        
        for driver in drivers:
            team = self.driver_team_mapping.get(driver, 'Unknown')
            if team == 'Unknown':
                continue
            
            # Create feature vector for this driver
            feature_vector = self._create_feature_vector(
                driver, team, race_name, circuit_info
            )
            
            # Predict using ML model
            try:
                predicted_position = self.model.predict(feature_vector)[0]
                
                # Get uncertainty estimate
                if hasattr(self.model, 'predict_with_uncertainty'):
                    mean_pred, std_pred = self.model.predict_with_uncertainty(feature_vector)
                    uncertainty = std_pred[0]
                else:
                    uncertainty = 1.5  # Default uncertainty
                
                predictions.append({
                    'driver': driver,
                    'team': team,
                    'raw_predicted_position': float(predicted_position),
                    'uncertainty': float(uncertainty)
                })
                
            except Exception as e:
                print(f"  Warning: Error predicting for {driver}: {e}")
                continue
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['raw_predicted_position'])
        
        # Assign unique integer positions
        for rank, pred in enumerate(predictions, 1):
            pred['predicted_position'] = rank
            pred['predicted_points'] = self._position_to_points(rank)
            pred['win_probability'] = self._calculate_win_probability(
                pred['raw_predicted_position'], 
                pred['uncertainty']
            )
            # Clean up
            del pred['raw_predicted_position']
            del pred['uncertainty']
        
        return {
            'race_name': race_name,
            'predictions': predictions[:10],
            'winner': predictions[0] if predictions else {},
            'podium': predictions[:3] if len(predictions) >= 3 else predictions,
            'confidence': 'High' if self.model.best_model_name == 'Ensemble' else 'Medium',
            'model_used': self.model.best_model_name
        }
    
    def _create_feature_vector(self, driver, team, race_name, circuit_info):
        """
        Create a complete feature vector for prediction
        Uses historical data and circuit information
        """
        # Get driver's historical data
        if self.historical_data is not None:
            driver_history = self.historical_data[
                self.historical_data['Driver'] == driver
            ].tail(10)  # Last 10 races
        else:
            driver_history = pd.DataFrame()
        
        # Create base features
        features = {}
        
        # Basic encodings
        features['Driver_Encoded'] = self.label_encoders['Driver'].transform([driver])[0]
        features['Team_Encoded'] = self.label_encoders['Team'].transform([team])[0]
        features['Race_Encoded'] = self.label_encoders['Race'].transform([race_name])[0] if race_name in self.label_encoders['Race'].classes_ else 0
        
        # Grid position (estimate from recent form)
        features['GridPosition'] = self._estimate_grid_position(driver, team, driver_history)
        
        # Historical features
        features['DriverCircuitAvg'] = self._get_circuit_average(driver, race_name, driver_history)
        features['TeamCircuitAvg'] = 10.0  # Default
        features['DriverCircuitBest'] = 8.0  # Default
        
        # Recent form features
        if len(driver_history) >= 3:
            features['RecentForm3'] = driver_history['Position'].tail(3).mean()
            features['RecentForm5'] = driver_history['Position'].tail(5).mean()
            features['RecentPoints3'] = driver_history['Points'].tail(3).sum()
            features['FormTrend'] = driver_history['Position'].tail(3).iloc[0] - driver_history['Position'].tail(3).iloc[-1]
        else:
            features['RecentForm3'] = 10.0
            features['RecentForm5'] = 10.0
            features['RecentPoints3'] = 0.0
            features['FormTrend'] = 0.0
        
        # Tire features (estimate)
        features['TireDegradation'] = 1.0
        features['PitStopImpact'] = 22.0
        features['OptimalStrategy'] = 1
        features['CircuitTireWear'] = 1.0
        features['TireCircuitInteraction'] = 1.0
        features['TireCompound_Encoded'] = 1  # Medium
        
        # Circuit features
        features['CircuitType_Encoded'] = 1 if circuit_info['type'] == 'street' else 0
        features['NormalizedCircuitLength'] = circuit_info['length'] / 7.0
        features['NormalizedCorners'] = circuit_info['corners'] / 23.0
        features['NormalizedSpeed'] = circuit_info['avg_speed'] / 270.0
        features['OvertakingFactor'] = 11 - circuit_info['overtaking_difficulty']
        features['CircuitComplexity'] = circuit_info['corners'] * 0.4 + circuit_info['overtaking_difficulty'] * 0.3
        
        # Weather features (estimate)
        features['AirTemp'] = 25.0
        features['TrackTemp'] = 35.0
        features['Humidity'] = 50.0
        features['Rainfall'] = 0
        features['TempTireInteraction'] = 35.0
        features['WeatherTireStress'] = 0.7
        features['HumidityEffect'] = 0.5
        
        # Championship features (estimate)
        features['ChampionshipPosition'] = 5.0
        features['PointsGapToLeader'] = 50.0
        features['TotalPressure'] = 5.0
        
        # Qualifying features (estimate)
        features['GapToPole'] = 0.5
        features['QualiVsTeammate'] = 0.0
        features['GridVsTeamAvg'] = 0.0
        
        # Team dynamics
        features['TeamRecentForm'] = 8.0
        features['TeamReliability'] = 0.95
        features['FasterThanTeammate'] = 1
        
        # Temporal features
        features['SeasonProgression'] = 0.8  # Late season
        features['DriverExperience'] = 100  # Estimate
        
        # Convert to DataFrame with proper column order
        feature_columns = self.preprocessor.get_feature_columns()
        feature_df = pd.DataFrame([features])
        
        # Ensure all columns exist, fill missing with defaults
        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Return in correct order
        return feature_df[feature_columns]
    
    def _get_circuit_info(self, race_name):
        """Get circuit characteristics"""
        circuit_db = {
            'Las Vegas Grand Prix': {
                'type': 'street', 'length': 6.201, 'corners': 17,
                'avg_speed': 240, 'overtaking_difficulty': 5
            },
            'Qatar Grand Prix': {
                'type': 'permanent', 'length': 5.380, 'corners': 16,
                'avg_speed': 230, 'overtaking_difficulty': 6
            },
            'Abu Dhabi Grand Prix': {
                'type': 'permanent', 'length': 5.281, 'corners': 16,
                'avg_speed': 195, 'overtaking_difficulty': 5
            }
        }
        
        return circuit_db.get(race_name, {
            'type': 'permanent', 'length': 5.0, 'corners': 16,
            'avg_speed': 200, 'overtaking_difficulty': 5
        })
    
    def _estimate_grid_position(self, driver, team, driver_history):
        """Estimate grid position from team strength and recent form"""
        team_positions = {
            'McLaren': 2, 'Ferrari': 3, 'Red Bull Racing': 4,
            'Mercedes': 6, 'Aston Martin': 9, 'Alpine': 12,
            'Haas F1 Team': 13, 'RB': 15, 'Williams': 17, 'Kick Sauber': 19
        }
        
        base = team_positions.get(team, 10)
        
        # Adjust based on recent form
        if len(driver_history) >= 3:
            recent_avg = driver_history['GridPosition'].tail(3).mean()
            base = int((base + recent_avg) / 2)
        
        return max(1, min(20, base))
    
    def _get_circuit_average(self, driver, race_name, driver_history):
        """Get driver's historical average at this circuit"""
        if len(driver_history) == 0:
            return 10.0
        
        circuit_races = driver_history[driver_history['Race'] == race_name]
        
        if len(circuit_races) > 0:
            return circuit_races['Position'].mean()
        else:
            return driver_history['Position'].mean()
    
    def _position_to_points(self, position):
        """Convert position to F1 points"""
        points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        return points.get(position, 0)
    
    def _calculate_win_probability(self, predicted_position, uncertainty):
        """
        Calculate win probability based on predicted position and uncertainty
        """
        if predicted_position <= 1.0:
            base_prob = 90.0
        elif predicted_position <= 1.5:
            base_prob = 70.0
        elif predicted_position <= 2.0:
            base_prob = 50.0
        elif predicted_position <= 3.0:
            base_prob = 30.0
        elif predicted_position <= 5.0:
            base_prob = 15.0
        else:
            base_prob = 5.0
        
        # Adjust for uncertainty (higher uncertainty = lower confidence)
        uncertainty_factor = max(0.5, 1.0 - (uncertainty / 5.0))
        
        return round(base_prob * uncertainty_factor, 1)