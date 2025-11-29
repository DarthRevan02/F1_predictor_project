"""
Enhanced Master Predictor for F1 Prediction System

This module coordinates all prediction modules and integrates with the enhanced ML models
"""

import numpy as np
import pandas as pd
from f1_predictor_project.prediction_modules.enhanced_prediction_race_winner import EnhancedRaceWinnerPredictor
from prediction_modules.prediction_podium import PodiumPredictor
from prediction_modules.prediction_qualifying import QualifyingPredictor
from prediction_modules.prediction_fastest_lap import FastestLapPredictor
from prediction_modules.prediction_points_finish import PointsFinishPredictor
from f1_predictor_project.prediction_modules.enhanced_prediction_dnf_probability import EnhancedDNFPredictor
from prediction_modules.prediction_overtakes import OvertakePredictor
from prediction_modules.prediction_championship_impact import ChampionshipImpactPredictor
from prediction_modules.prediction_team_performance import TeamPerformancePredictor
from prediction_modules.prediction_strategy import StrategyPredictor

class EnhancedMasterF1Predictor:
    """
    Enhanced Master predictor that provides comprehensive F1 race predictions
    Uses advanced ML models and engineered features
    """
    
    def __init__(self, model, preprocessor, label_encoders, current_standings=None, historical_data=None):
        self.model = model  # Enhanced ML model
        self.preprocessor = preprocessor  # Feature engineering pipeline
        self.label_encoders = label_encoders
        self.current_standings = current_standings or {}
        self.historical_data = historical_data
        
        # Initialize all specialized predictors with enhanced capabilities
        self.race_winner_predictor = EnhancedRaceWinnerPredictor(
            model, preprocessor, label_encoders, historical_data
        )
        self.podium_predictor = PodiumPredictor(model, label_encoders)
        self.qualifying_predictor = QualifyingPredictor(model, label_encoders)
        self.fastest_lap_predictor = FastestLapPredictor(model, label_encoders)
        self.points_predictor = PointsFinishPredictor(model, label_encoders)
        self.dnf_predictor = EnhancedDNFPredictor(model, label_encoders, historical_data)
        self.overtake_predictor = OvertakePredictor(model, label_encoders)
        self.championship_predictor = ChampionshipImpactPredictor(
            model, label_encoders, current_standings
        )
        self.team_predictor = TeamPerformancePredictor(model, label_encoders)
        self.strategy_predictor = StrategyPredictor(model, label_encoders)
    
    def get_comprehensive_predictions(self, race_name="Next Race", race_distance=58):
        """
        Get all predictions for a race using enhanced models
        """
        print(f"\n🏁 Generating enhanced predictions for {race_name}...")
        
        predictions = {
            'race_name': race_name,
            'model_info': {
                'model_type': self.model.best_model_name,
                'model_r2': float(self.model.model_scores[self.model.best_model_name]['cv_mean']),
                'model_mae': float(self.model.model_scores[self.model.best_model_name]['train_mae'])
            },
            'predictions': {}
        }
        
        # 1. Race Winner Prediction (Enhanced with ML model)
        print("  [1/10] Predicting race winner with advanced ML...")
        predictions['predictions']['race_winner'] = self.race_winner_predictor.predict_race_winner(race_name)
        
        # 2. Podium Prediction
        print("  [2/10] Analyzing podium probabilities...")
        predictions['predictions']['podium'] = self.podium_predictor.predict_podium(race_name)
        
        # 3. Qualifying Prediction
        print("  [3/10] Predicting qualifying results...")
        predictions['predictions']['qualifying'] = self.qualifying_predictor.predict_qualifying(race_name)
        
        # 4. Fastest Lap Prediction
        print("  [4/10] Predicting fastest lap...")
        predictions['predictions']['fastest_lap'] = self.fastest_lap_predictor.predict_fastest_lap(race_name)
        
        # 5. Points Finishers Prediction
        print("  [5/10] Analyzing points-scoring opportunities...")
        predictions['predictions']['points_finishers'] = self.points_predictor.predict_points_finishers(race_name)
        
        # 6. Enhanced DNF Probability Prediction
        print("  [6/10] Calculating DNF risks with historical data...")
        predictions['predictions']['dnf_risk'] = self.dnf_predictor.predict_dnf_risk(race_name)
        
        # 7. Overtaking Prediction
        print("  [7/10] Predicting overtaking action...")
        predictions['predictions']['overtakes'] = self.overtake_predictor.predict_position_changes(race_name)
        
        # 8. Championship Impact Prediction
        print("  [8/10] Analyzing championship impact...")
        predictions['predictions']['championship_impact'] = self.championship_predictor.predict_championship_impact(race_name)
        
        # 9. Team Performance Prediction
        print("  [9/10] Evaluating team performance...")
        predictions['predictions']['team_performance'] = self.team_predictor.predict_team_performance(race_name)
        
        # 10. Strategy Prediction
        print("  [10/10] Predicting race strategies...")
        predictions['predictions']['strategy'] = self.strategy_predictor.predict_race_strategy(race_name, race_distance)
        
        print("✓ All enhanced predictions generated!\n")
        
        return predictions
    
    def get_prediction_summary(self, race_name="Next Race"):
        """
        Get a concise summary of key predictions
        """
        comprehensive = self.get_comprehensive_predictions(race_name)
        
        summary = {
            'race_name': race_name,
            'model_confidence': f"{comprehensive['model_info']['model_r2']:.1%}",
            'predicted_winner': comprehensive['predictions']['race_winner']['winner']['driver'],
            'win_probability': comprehensive['predictions']['race_winner']['winner']['win_probability'],
            'predicted_podium': [
                p['driver'] for p in comprehensive['predictions']['race_winner']['podium']
            ],
            'pole_position': comprehensive['predictions']['qualifying']['pole_position']['driver'],
            'fastest_lap_favorite': comprehensive['predictions']['fastest_lap']['most_likely']['driver'],
            'top_team': comprehensive['predictions']['team_performance']['top_team']['team'],
            'highest_dnf_risk': comprehensive['predictions']['dnf_risk']['highest_risk'][0]['driver'],
            'best_overtaker': comprehensive['predictions']['overtakes']['best_overtakers'][0]['driver']
        }
        
        return summary