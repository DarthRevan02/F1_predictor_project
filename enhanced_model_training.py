import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class EnhancedF1RaceModel:
    """
    Advanced ML models with ensemble methods and cross-validation
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.ensemble_model = None
        self.model_scores = {}
        self.is_trained = False
        
    def train_all_models(self, X, y, use_ensemble=True):
        """
        Train multiple models and compare performance
        """
        print("\n" + "="*60)
        print("🤖 ADVANCED MODEL TRAINING")
        print("="*60)
        
        print(f"\n📊 Training dataset:")
        print(f"   - Samples: {len(X)}")
        print(f"   - Features: {len(X.columns)}")
        print(f"   - Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Initialize models
        self._initialize_models()
        
        print(f"\n🔧 Training {len(self.models)} different models...")
        print(f"   Using Time Series Cross-Validation (5 folds)\n")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_score = -float('inf')
        
        for name, model in self.models.items():
            print(f"{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=tscv, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"   Cross-Validation R² Scores: {cv_scores}")
                print(f"   Mean R²: {cv_mean:.4f} (+/- {cv_std:.4f})")
                
                # Train on full dataset
                model.fit(X, y)
                
                # Evaluate on training set
                y_pred = model.predict(X)
                train_r2 = r2_score(y, y_pred)
                train_mae = mean_absolute_error(y, y_pred)
                train_rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                print(f"   Training R²: {train_r2:.4f}")
                print(f"   Training MAE: {train_mae:.4f} positions")
                print(f"   Training RMSE: {train_rmse:.4f} positions")
                
                # Store model and scores
                self.model_scores[name] = {
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'train_r2': train_r2,
                    'train_mae': train_mae,
                    'train_rmse': train_rmse
                }
                
                # Track best model
                if cv_mean > best_score:
                    best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"   ✓ {name} training complete\n")
                
            except Exception as e:
                print(f"   ✗ Error training {name}: {e}\n")
                continue
        
        # Create ensemble if requested
        if use_ensemble and len(self.models) > 1:
            self._create_ensemble_model(X, y, tscv)
        
        self.is_trained = True
        
        # Print final results
        self._print_training_summary()
        
        return self.model_scores
    
    def _initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
    
    def _create_ensemble_model(self, X, y, tscv):
        """Create weighted ensemble model"""
        print(f"{'='*60}")
        print(f"Creating Ensemble Model (Weighted Voting)")
        print(f"{'='*60}")
        
        # Calculate weights based on CV performance
        weights = []
        estimators = []
        
        for name, model in self.models.items():
            if name in self.model_scores:
                score = self.model_scores[name]['cv_mean']
                if score > 0:  # Only use models with positive R²
                    weight = max(0.1, score)  # Minimum weight of 0.1
                    weights.append(weight)
                    estimators.append((name, model))
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"\n   Ensemble composition:")
        for (name, _), weight in zip(estimators, weights):
            print(f"   - {name}: {weight:.2%}")
        
        # Create voting regressor
        self.ensemble_model = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )
        
        # Evaluate ensemble
        cv_scores = cross_val_score(
            self.ensemble_model, X, y,
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n   Ensemble CV R²: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Train ensemble
        self.ensemble_model.fit(X, y)
        
        y_pred = self.ensemble_model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_mae = mean_absolute_error(y, y_pred)
        
        print(f"   Ensemble Training R²: {train_r2:.4f}")
        print(f"   Ensemble Training MAE: {train_mae:.4f} positions")
        
        self.model_scores['Ensemble'] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_r2': train_r2,
            'train_mae': train_mae
        }
        
        # Update best model if ensemble is better
        if cv_mean > self.model_scores[self.best_model_name]['cv_mean']:
            self.best_model = self.ensemble_model
            self.best_model_name = 'Ensemble'
            print(f"\n   🏆 Ensemble selected as best model!")
        
        print(f"   ✓ Ensemble model created\n")
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        
        # Sort models by CV score
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['cv_mean'],
            reverse=True
        )
        
        print("\n🏆 Model Rankings (by Cross-Validation R²):\n")
        
        for rank, (name, scores) in enumerate(sorted_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"{medal} {name}")
            print(f"   CV R²:        {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")
            print(f"   Training R²:  {scores['train_r2']:.4f}")
            print(f"   MAE:          {scores['train_mae']:.2f} positions")
            if 'train_rmse' in scores:
                print(f"   RMSE:         {scores['train_rmse']:.2f} positions")
            print()
        
        print(f"{'='*60}")
        print(f"🎯 BEST MODEL: {self.best_model_name}")
        print(f"   R² Score: {self.model_scores[self.best_model_name]['cv_mean']:.4f}")
        print(f"   MAE: {self.model_scores[self.best_model_name]['train_mae']:.2f} positions")
        print(f"{'='*60}\n")
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.best_model.predict(X)
        
        # Ensure predictions are within valid range (1-20)
        predictions = np.clip(predictions, 1, 20)
        
        return predictions
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimates using all models
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        all_predictions = []
        
        for name, model in self.models.items():
            if name in self.model_scores:
                pred = model.predict(X)
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        mean_prediction = all_predictions.mean(axis=0)
        std_prediction = all_predictions.std(axis=0)
        
        return mean_prediction, std_prediction
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from tree-based models
        """
        importance_dict = {}
        
        # Random Forest importance
        if 'Random Forest' in self.models and hasattr(self.models['Random Forest'], 'feature_importances_'):
            rf_importance = self.models['Random Forest'].feature_importances_
            importance_dict['Random Forest'] = dict(zip(feature_names, rf_importance))
        
        # Gradient Boosting importance
        if 'Gradient Boosting' in self.models and hasattr(self.models['Gradient Boosting'], 'feature_importances_'):
            gb_importance = self.models['Gradient Boosting'].feature_importances_
            importance_dict['Gradient Boosting'] = dict(zip(feature_names, gb_importance))
        
        if importance_dict:
            print("\n" + "="*60)
            print("📊 TOP 15 MOST IMPORTANT FEATURES")
            print("="*60)
            
            for model_name, importances in importance_dict.items():
                print(f"\n{model_name}:")
                sorted_features = sorted(
                    importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                
                for rank, (feature, importance) in enumerate(sorted_features, 1):
                    bar = "█" * int(importance * 50)
                    print(f"  {rank:2d}. {feature:30s} {bar} {importance:.4f}")
            
            print("="*60 + "\n")
        
        return importance_dict
    
    def save_model(self, filepath):
        """Save the best model"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'model_scores': self.model_scores
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.best_model = data['best_model']
            self.best_model_name = data['best_model_name']
            self.model_scores = data['model_scores']
            self.is_trained = True
        print(f"✓ Model loaded from {filepath}")