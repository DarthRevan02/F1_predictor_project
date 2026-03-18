from sklearn.linear_model import LinearRegression
import numpy as np


class F1RaceModel:
    """Trains a linear regression model to predict finishing positions."""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X, y):
        """
        Train the model.  Raises ValueError on empty input so the caller
        (initialize_system) surfaces the problem clearly — fixes flaw #9.
        """
        if len(X) == 0:
            raise ValueError("Cannot train: feature matrix X is empty.")

        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_trained = True

        score = self.model.score(X, y)
        print(f"✓ Model trained on {len(X)} samples — R² = {score:.3f}")
        print(f"✓ Coefficients: {self.model.coef_}")
        return score

    def predict(self, X):
        """
        Predict finishing position.

        Always returns a plain Python int (fixes flaw #10 — numpy int vs
        Python int ambiguity, and double-rounding in the route handler).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")

        raw = self.model.predict(X)
        # raw is a 1-D numpy array; grab first element, clip to valid range
        value = float(raw[0]) if hasattr(raw, '__len__') else float(raw)
        return int(max(1, round(value)))
