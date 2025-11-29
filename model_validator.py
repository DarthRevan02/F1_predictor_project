import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ModelValidator:
    """
    Comprehensive model validation and backtesting
    """
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        
    def backtest_predictions(self, historical_data, test_year=2024):
        """
        Test model predictions against actual race results
        """
        print("\n" + "="*60)
        print("🔍 MODEL BACKTESTING & VALIDATION")
        print("="*60)
        
        # Split data: train on years before test_year, test on test_year
        train_data = historical_data[historical_data['Year'] < test_year]
        test_data = historical_data[historical_data['Year'] == test_year]
        
        if len(test_data) == 0:
            print(f"⚠️  No test data available for {test_year}")
            return None
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training: {len(train_data)} samples ({train_data['Year'].min()}-{train_data['Year'].max()})")
        print(f"   Testing:  {len(test_data)} samples ({test_year})")
        
        # Get feature columns
        feature_cols = self.preprocessor.get_feature_columns()
        
        # Prepare test data
        X_test = test_data[feature_cols]
        y_test = test_data['Position']
        
        # Make predictions
        print(f"\n🎯 Making predictions on {len(test_data)} test samples...")
        y_pred = self.model.predict(X_test)
        
        # Round predictions to nearest integer position
        y_pred_rounded = np.round(y_pred).astype(int)
        y_pred_rounded = np.clip(y_pred_rounded, 1, 20)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Custom metrics
        exact_matches = (y_pred_rounded == y_test).sum()
        within_1_position = (np.abs(y_pred_rounded - y_test) <= 1).sum()
        within_2_positions = (np.abs(y_pred_rounded - y_test) <= 2).sum()
        within_3_positions = (np.abs(y_pred_rounded - y_test) <= 3).sum()
        
        total_predictions = len(y_test)
        
        print(f"\n{'='*60}")
        print(f"📈 OVERALL METRICS")
        print(f"{'='*60}")
        print(f"   R² Score:           {r2:.4f}")
        print(f"   Mean Absolute Error: {mae:.2f} positions")
        print(f"   Root Mean Squared:   {rmse:.2f} positions")
        print(f"\n   Accuracy:")
        print(f"   - Exact position:    {exact_matches}/{total_predictions} ({exact_matches/total_predictions*100:.1f}%)")
        print(f"   - Within ±1 pos:     {within_1_position}/{total_predictions} ({within_1_position/total_predictions*100:.1f}%)")
        print(f"   - Within ±2 pos:     {within_2_positions}/{total_predictions} ({within_2_positions/total_predictions*100:.1f}%)")
        print(f"   - Within ±3 pos:     {within_3_positions}/{total_predictions} ({within_3_positions/total_predictions*100:.1f}%)")
        
        # Race-by-race analysis
        print(f"\n{'='*60}")
        print(f"🏁 RACE-BY-RACE ANALYSIS")
        print(f"{'='*60}\n")
        
        race_results = self._analyze_by_race(test_data, y_pred_rounded)
        
        # Winner prediction accuracy
        winner_accuracy = self._analyze_winner_predictions(test_data, y_pred_rounded)
        
        # Podium prediction accuracy
        podium_accuracy = self._analyze_podium_predictions(test_data, y_pred_rounded)
        
        # Top 10 prediction accuracy
        top10_accuracy = self._analyze_top10_predictions(test_data, y_pred_rounded)
        
        print(f"\n{'='*60}")
        print(f"🎯 PREDICTION ACCURACY BY CATEGORY")
        print(f"{'='*60}")
        print(f"   Winner Predictions:  {winner_accuracy:.1f}%")
        print(f"   Podium Predictions:  {podium_accuracy:.1f}%")
        print(f"   Top 10 Predictions:  {top10_accuracy:.1f}%")
        print(f"{'='*60}\n")
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'exact_accuracy': exact_matches / total_predictions * 100,
            'within_1_accuracy': within_1_position / total_predictions * 100,
            'within_2_accuracy': within_2_positions / total_predictions * 100,
            'within_3_accuracy': within_3_positions / total_predictions * 100,
            'winner_accuracy': winner_accuracy,
            'podium_accuracy': podium_accuracy,
            'top10_accuracy': top10_accuracy,
            'race_results': race_results
        }
    
    def _analyze_by_race(self, test_data, predictions):
        """
        Analyze predictions for each race
        """
        test_data_copy = test_data.copy()
        test_data_copy['Predicted'] = predictions
        
        race_results = []
        
        for race in test_data_copy['Race'].unique():
            race_data = test_data_copy[test_data_copy['Race'] == race]
            
            actual = race_data['Position'].values
            predicted = race_data['Predicted'].values
            
            # Calculate race-specific metrics
            mae = mean_absolute_error(actual, predicted)
            
            # Winner prediction
            actual_winner = race_data[race_data['Position'] == 1]['Driver'].values[0]
            predicted_winner = race_data[race_data['Predicted'] == 1]['Driver'].values
            winner_correct = actual_winner in predicted_winner if len(predicted_winner) > 0 else False
            
            # Podium prediction
            actual_podium = set(race_data[race_data['Position'] <= 3]['Driver'].values)
            predicted_podium = set(race_data[race_data['Predicted'] <= 3]['Driver'].values)
            podium_overlap = len(actual_podium & predicted_podium)
            
            race_results.append({
                'race': race,
                'mae': mae,
                'winner_correct': winner_correct,
                'podium_overlap': podium_overlap
            })
            
            winner_emoji = "✅" if winner_correct else "❌"
            print(f"   {winner_emoji} {race:40s} MAE: {mae:.2f}  Podium: {podium_overlap}/3")
        
        return race_results
    
    def _analyze_winner_predictions(self, test_data, predictions):
        """
        Calculate winner prediction accuracy
        """
        test_data_copy = test_data.copy()
        test_data_copy['Predicted'] = predictions
        
        correct_winners = 0
        total_races = 0
        
        for race in test_data_copy['Race'].unique():
            race_data = test_data_copy[test_data_copy['Race'] == race]
            
            actual_winner = race_data[race_data['Position'] == 1]['Driver'].values[0]
            predicted_winner_data = race_data[race_data['Predicted'] == 1]
            
            if len(predicted_winner_data) > 0:
                predicted_winner = predicted_winner_data['Driver'].values[0]
                if actual_winner == predicted_winner:
                    correct_winners += 1
            
            total_races += 1
        
        return (correct_winners / total_races * 100) if total_races > 0 else 0
    
    def _analyze_podium_predictions(self, test_data, predictions):
        """
        Calculate podium prediction accuracy (all 3 correct)
        """
        test_data_copy = test_data.copy()
        test_data_copy['Predicted'] = predictions
        
        correct_podiums = 0
        total_races = 0
        
        for race in test_data_copy['Race'].unique():
            race_data = test_data_copy[test_data_copy['Race'] == race]
            
            actual_podium = set(race_data[race_data['Position'] <= 3]['Driver'].values)
            predicted_podium = set(race_data[race_data['Predicted'] <= 3]['Driver'].values)
            
            if actual_podium == predicted_podium:
                correct_podiums += 1
            
            total_races += 1
        
        return (correct_podiums / total_races * 100) if total_races > 0 else 0
    
    def _analyze_top10_predictions(self, test_data, predictions):
        """
        Calculate top 10 prediction accuracy
        """
        test_data_copy = test_data.copy()
        test_data_copy['Predicted'] = predictions
        
        correct_drivers_in_top10 = 0
        total_opportunities = 0
        
        for race in test_data_copy['Race'].unique():
            race_data = test_data_copy[test_data_copy['Race'] == race]
            
            actual_top10 = set(race_data[race_data['Position'] <= 10]['Driver'].values)
            predicted_top10 = set(race_data[race_data['Predicted'] <= 10]['Driver'].values)
            
            overlap = len(actual_top10 & predicted_top10)
            correct_drivers_in_top10 += overlap
            total_opportunities += 10
        
        return (correct_drivers_in_top10 / total_opportunities * 100) if total_opportunities > 0 else 0
    
    def cross_validate_by_season(self, historical_data):
        """
        Perform leave-one-season-out cross-validation
        """
        print("\n" + "="*60)
        print("🔄 CROSS-VALIDATION BY SEASON")
        print("="*60)
        
        years = sorted(historical_data['Year'].unique())
        cv_results = []
        
        for test_year in years:
            if test_year == years[0]:  # Skip first year (need training data)
                continue
            
            print(f"\n📅 Testing on {test_year}...")
            
            train_data = historical_data[historical_data['Year'] < test_year]
            test_data = historical_data[historical_data['Year'] == test_year]
            
            if len(test_data) < 10:  # Skip if too few test samples
                continue
            
            feature_cols = self.preprocessor.get_feature_columns()
            
            X_train = train_data[feature_cols]
            y_train = train_data['Position']
            X_test = test_data[feature_cols]
            y_test = test_data['Position']
            
            # Train and predict
            self.model.best_model.fit(X_train, y_train)
            y_pred = self.model.best_model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_results.append({
                'year': test_year,
                'mae': mae,
                'r2': r2,
                'n_samples': len(test_data)
            })
            
            print(f"   MAE: {mae:.2f}, R²: {r2:.4f}, Samples: {len(test_data)}")
        
        # Summary
        avg_mae = np.mean([r['mae'] for r in cv_results])
        avg_r2 = np.mean([r['r2'] for r in cv_results])
        
        print(f"\n{'='*60}")
        print(f"📊 CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"   Average MAE: {avg_mae:.2f} positions")
        print(f"   Average R²:  {avg_r2:.4f}")
        print(f"   Years tested: {len(cv_results)}")
        print(f"{'='*60}\n")
        
        return cv_results
    
    def analyze_prediction_errors(self, historical_data, test_year=2024):
        """
        Detailed error analysis
        """
        print("\n" + "="*60)
        print("🔬 PREDICTION ERROR ANALYSIS")
        print("="*60)
        
        test_data = historical_data[historical_data['Year'] == test_year]
        
        if len(test_data) == 0:
            print(f"⚠️  No data for {test_year}")
            return
        
        feature_cols = self.preprocessor.get_feature_columns()
        X_test = test_data[feature_cols]
        y_test = test_data['Position']
        
        y_pred = self.model.predict(X_test)
        errors = y_test - y_pred
        
        print(f"\n📊 Error Distribution:")
        print(f"   Mean Error:     {errors.mean():.2f} positions")
        print(f"   Std Dev:        {errors.std():.2f} positions")
        print(f"   Max Overpredict: {errors.min():.2f} positions")
        print(f"   Max Underpredict: {errors.max():.2f} positions")
        
        # Error by position
        print(f"\n📍 Error by Grid Position:")
        for grid_group in [(1, 5), (6, 10), (11, 15), (16, 20)]:
            mask = (test_data['GridPosition'] >= grid_group[0]) & (test_data['GridPosition'] <= grid_group[1])
            group_errors = errors[mask]
            if len(group_errors) > 0:
                print(f"   Grid P{grid_group[0]:2d}-P{grid_group[1]:2d}: MAE = {np.abs(group_errors).mean():.2f}")
        
        # Error by driver
        print(f"\n👤 Error by Driver (Top 10 most predicted):")
        driver_errors = test_data.copy()
        driver_errors['Error'] = errors
        driver_error_summary = driver_errors.groupby('Driver')['Error'].agg(['count', 'mean', 'std'])
        driver_error_summary = driver_error_summary.sort_values('count', ascending=False).head(10)
        
        for driver, row in driver_error_summary.iterrows():
            print(f"   {driver}: Count={int(row['count']):2d}, Mean Error={row['mean']:+.2f}, Std={row['std']:.2f}")
        
        print(f"\n{'='*60}\n")