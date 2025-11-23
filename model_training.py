from sklearn.linear_model import LinearRegression
import numpy as np

class F1RaceModel:
    """Handles training and predictions for race positions"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def train(self, X, y):
        """
        Train the linear regression model
        
        Args:
            X: Feature matrix (Driver, Team, Grid Position)
            y: Target variable (Finishing Position)
        """
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate and print model performance
        score = self.model.score(X, y)
        print(f"✓ Model trained successfully!")
        print(f"✓ Model R² score: {score:.3f}")
        
        # Print coefficients for interpretation
        print(f"✓ Model coefficients: {self.model.coef_}")
        
        return score
    
    def predict(self, X):
        """
        Predict race finishing position
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted finishing position
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        prediction = self.model.predict(X)[0]
        return max(1, round(prediction))