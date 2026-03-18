import numpy as np
from driver_config import (
    DRIVER_TEAM_MAP, DRIVER_PERFORMANCE, TEAM_GRID_BASE, F1_POINTS_SYSTEM
)


class RaceWinnerPredictor:
    """Predicts race winner and top-10 using the trained ML model."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_race_winner(self, race_name="Next Race"):
        """
        Use the ML model to rank all known 2025 drivers.

        The model predicts a raw finishing score from
        (driver_encoded, team_encoded, estimated_grid).  Drivers are then
        ranked by that score and assigned unique integer positions.

        Previously the model output was collected but then re-sorted by a
        second hardcoded heuristic, making the ML irrelevant (flaw #3).
        That second sort has been removed — the model IS the predictor.
        """
        raw_predictions = []

        for driver, team in DRIVER_TEAM_MAP.items():
            estimated_grid = self._estimate_grid_position(team)
            raw_score = self._predict_raw_score(driver, team, estimated_grid)

            if raw_score is None:
                # Driver/team not in encoder — fall back to performance rating
                raw_score = (1 - DRIVER_PERFORMANCE.get(driver, 0.65)) * 20

            raw_predictions.append({
                'driver': driver,
                'team': team,
                'estimated_grid': estimated_grid,
                '_raw_score': raw_score,
            })

        # Sort by model output (lower predicted position = better finish)
        raw_predictions.sort(key=lambda x: x['_raw_score'])

        predictions = []
        for rank, pred in enumerate(raw_predictions, start=1):
            predictions.append({
                'driver': pred['driver'],
                'team': pred['team'],
                'estimated_grid': pred['estimated_grid'],
                'predicted_position': rank,
                'predicted_points': F1_POINTS_SYSTEM.get(rank, 0),
                'win_probability': self._win_probability(pred['_raw_score'], rank),
            })
            del pred['_raw_score']

        return {
            'race_name': race_name,
            'predictions': predictions[:10],
            'winner': predictions[0] if predictions else {},
            'podium': predictions[:3],
        }

    def _predict_raw_score(self, driver, team, grid_position):
        """
        Call the ML model.  Returns None if either label is unknown so the
        caller can fall back gracefully instead of crashing (flaw #8 guard).
        """
        try:
            driver_enc = self.label_encoders['Driver'].transform([driver])[0]
            team_enc = self.label_encoders['Team'].transform([team])[0]
        except ValueError:
            return None

        X = np.array([[driver_enc, team_enc, grid_position]])
        raw = self.model.model.predict(X)
        return float(raw[0]) if hasattr(raw, '__len__') else float(raw)

    def _estimate_grid_position(self, team):
        """Estimate qualifying grid from team baseline + small random variance."""
        base = TEAM_GRID_BASE.get(team, 10)
        return max(1, min(20, base + int(np.random.randint(-2, 3))))

    def _win_probability(self, raw_score, rank):
        """Convert model raw score to a rough win-probability percentage."""
        if rank == 1:
            return round(min(95.0, max(10.0, 100 - raw_score * 5)), 1)
        return round(max(0.0, 40 - (rank - 2) * 8), 1)
