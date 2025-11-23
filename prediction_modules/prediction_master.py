from prediction_modules.prediction_race_winner import RaceWinnerPredictor
from prediction_modules.prediction_podium import PodiumPredictor
from prediction_modules.prediction_qualifying import QualifyingPredictor
from prediction_modules.prediction_fastest_lap import FastestLapPredictor
from prediction_modules.prediction_points_finish import PointsFinishPredictor
from prediction_modules.prediction_dnf_probability import DNFPredictor
from prediction_modules.prediction_overtakes import OvertakePredictor
from prediction_modules.prediction_championship_impact import ChampionshipImpactPredictor
from prediction_modules.prediction_team_performance import TeamPerformancePredictor
from prediction_modules.prediction_strategy import StrategyPredictor

class MasterF1Predictor:
    """
    Master predictor that provides comprehensive F1 race predictions
    using all specialized prediction modules
    """
    
    def __init__(self, model, label_encoders, current_standings=None):
        self.model = model
        self.label_encoders = label_encoders
        self.current_standings = current_standings or {}
        
        # Initialize all specialized predictors
        self.race_winner_predictor = RaceWinnerPredictor(model, label_encoders)
        self.podium_predictor = PodiumPredictor(model, label_encoders)
        self.qualifying_predictor = QualifyingPredictor(model, label_encoders)
        self.fastest_lap_predictor = FastestLapPredictor(model, label_encoders)
        self.points_predictor = PointsFinishPredictor(model, label_encoders)
        self.dnf_predictor = DNFPredictor(model, label_encoders)
        self.overtake_predictor = OvertakePredictor(model, label_encoders)
        self.championship_predictor = ChampionshipImpactPredictor(model, label_encoders, current_standings)
        self.team_predictor = TeamPerformancePredictor(model, label_encoders)
        self.strategy_predictor = StrategyPredictor(model, label_encoders)
    
    def get_comprehensive_predictions(self, race_name="Next Race", race_distance=58):
        """
        Get all predictions for a race
        
        Returns:
            dict: Comprehensive predictions from all modules
        """
        print(f"\n🏁 Generating comprehensive predictions for {race_name}...")
        
        predictions = {
            'race_name': race_name,
            'predictions': {}
        }
        
        # 1. Race Winner Prediction
        print("  [1/10] Predicting race winner...")
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
        
        # 6. DNF Probability Prediction
        print("  [6/10] Calculating DNF risks...")
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
        
        print("✓ All predictions generated!\n")
        
        return predictions
    
    def get_prediction_summary(self, race_name="Next Race"):
        """
        Get a concise summary of key predictions
        
        Returns:
            dict: Summary of most important predictions
        """
        comprehensive = self.get_comprehensive_predictions(race_name)
        
        summary = {
            'race_name': race_name,
            'predicted_winner': comprehensive['predictions']['race_winner']['winner']['driver'],
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
    
    def compare_drivers(self, driver1, driver2, race_name="Next Race"):
        """
        Compare two drivers head-to-head for a race
        
        Returns:
            dict: Detailed comparison
        """
        predictions = self.get_comprehensive_predictions(race_name)
        
        comparison = {
            'race_name': race_name,
            'driver1': driver1,
            'driver2': driver2,
            'comparison': {}
        }
        
        # Extract relevant metrics for both drivers
        for driver in [driver1, driver2]:
            driver_metrics = {
                'predicted_position': 'N/A',
                'podium_probability': 0,
                'quali_position': 'N/A',
                'dnf_risk': 0,
                'overtaking_rating': 0
            }
            
            # Get data from predictions
            for pred in predictions['predictions']['race_winner']['predictions']:
                if pred['driver'] == driver:
                    driver_metrics['predicted_position'] = pred['predicted_position']
            
            for pred in predictions['predictions']['podium']['top_contenders']:
                if pred['driver'] == driver:
                    driver_metrics['podium_probability'] = pred['podium_probability']
            
            for pred in predictions['predictions']['qualifying']['predicted_grid']:
                if pred['driver'] == driver:
                    driver_metrics['quali_position'] = pred['predicted_grid']
            
            for pred in predictions['predictions']['dnf_risk']['predictions']:
                if pred['driver'] == driver:
                    driver_metrics['dnf_risk'] = pred['dnf_probability']
            
            for pred in predictions['predictions']['overtakes']['predictions']:
                if pred['driver'] == driver:
                    driver_metrics['overtaking_rating'] = pred['overtaking_rating']
            
            comparison['comparison'][driver] = driver_metrics
        
        # Determine who has the advantage
        comparison['advantage'] = self._determine_advantage(
            comparison['comparison'][driver1],
            comparison['comparison'][driver2]
        )
        
        return comparison
    
    def _determine_advantage(self, driver1_metrics, driver2_metrics):
        """Determine which driver has the advantage"""
        score1 = 0
        score2 = 0
        
        if driver1_metrics['predicted_position'] != 'N/A' and driver2_metrics['predicted_position'] != 'N/A':
            if driver1_metrics['predicted_position'] < driver2_metrics['predicted_position']:
                score1 += 1
            else:
                score2 += 1
        
        if driver1_metrics['podium_probability'] > driver2_metrics['podium_probability']:
            score1 += 1
        else:
            score2 += 1
        
        if score1 > score2:
            return driver1_metrics
        else:
            return driver2_metrics
    def predict_race_winner(self, race_name="Next Race"):
     """
        Return only the predicted race winner (simple standalone API)
        """
     return {
        "race_name": race_name,
        "winner": self.race_winner_predictor.predict_race_winner(race_name)
    }
